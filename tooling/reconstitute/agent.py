"""Per-port lift agent: turn a standalone gemmi port into a `<class>_gemmi` method.

The phase-1 gemmi port is a free function `coot::foo_gemmi(gemmi::Structure& st,
args)` with a passing test. This agent rewrites it as a method of the parallel
class (`coot::<class>_gemmi::foo(args)` operating on the `structure` member) and
adapts the port's own test to construct the class and call the method — keeping the
frozen assertions. The lift is verified by compiling + running that adapted test.

We deliberately reuse the gemmi agent's low-level harness (compile/run/write tool
handlers, the Ollama chat client, block extraction) instead of duplicating it. The
loop here is much smaller than the port loop because the gemmi code already works —
no MMDB→gemmi API discovery is needed, only a mechanical re-homing the model can
verify by compiling.
"""
from __future__ import annotations

import json
import sqlite3
import textwrap
from pathlib import Path

from ..oracle.agent import _TraceWriter, _chat, _log_llm_timing
from .agent_reexports import (
    make_tool_handlers,
    extract_blocks,
    dep_extra_includes,
    dep_extra_sources,
    LIFT_TOOLS,
)
from .classmodel import ClassModel

DEFAULT_MODEL = "qwen3.6"
MAX_TURNS = 30


LIFT_SYSTEM_PROMPT = """\
You are RE-HOMING one already-working gemmi C++ function into a method of a class,
and adapting its Google Test. The gemmi function compiles and its test passes —
your job is a mechanical move, NOT a rewrite. Do not change the algorithm.

# Artifacts to produce
A. function.hh — `#pragma once`, the `#include <gemmi/...>` deps, and the class
   definition containing your method (inline body is fine). The whole class lives
   here so the test can construct it.
B. function.cc — OPTIONAL. Only if you prefer the method body out-of-line.
C. test.cc — the original gemmi test, adapted to construct the class object and
   call the method. Must `#include "function.hh"`.

# What changes (and ONLY this)
1. The free function's FIRST parameter is the model data — usually an explicit
   `gemmi::Structure&` (named st / mol / structure / ...), but sometimes a VIEW of
   it such as `gemmi::Model const&` or `gemmi::Model&`. As a method, DROP that
   parameter and obtain the equivalent from the `structure` member:
     - `gemmi::Structure&`     → use `structure` (or `this->structure`)
     - `gemmi::Model const&`   → use `structure.models[0]` (or a `const`/ref local
                                  bound to it), exactly where the param was used
   Substitute ONLY that one parameter; touch nothing else in the body.
2. Drop the `_gemmi` suffix from the function name; it becomes the method name.
3. ⚠ CRITICAL — COPY THE BODY VERBATIM. This is a mechanical re-homing, not a
   rewrite. Keep the other parameters, the return type, every statement, every
   loop, and every call EXACTLY as in the verified port. Do NOT:
     - shorten or "simplify" the body, even if it looks long or redundant;
     - replace a block of code with a call to some method you think is equivalent
       (e.g. swapping a hand-written loop that returns `result.overlaps.size()`
       for a `result.score()` call) — that method may not exist and will fail;
     - call ANY function or member that did not appear in the original port.
   The verified port already compiles and passes. The ONLY edits allowed are the
   first-parameter substitution (rule 1) and the name/signature change (rule 2).
   You may keep calling sibling `coot::*_gemmi(...)` free functions verbatim; only
   the one function you are lifting becomes a method.
4. In the test, replace the free-function call `coot::foo_gemmi(st, args)` with
   `coot::<class>_gemmi m(st); ... m.foo(args) ...`. Keep EVERY `EXPECT_*` /
   `ASSERT_*` assertion and its expected value identical — they are frozen.

# Workflow — write_gemmi_file is the ONLY path
Write files to disk with **write_gemmi_file**, in this order:
  1. function.cc (only if you split the body out)
  2. function.hh
  3. test.cc      ← writing this triggers an automatic compile + run
Fix any compile/test failure by rewriting the affected file(s). If the compile log
is truncated, call get_compile_errors before guessing.

# Terminal condition
Done when the auto-compile after writing test.cc succeeds AND the gtest output says
all tests passed. Then emit ONE final response with these fenced blocks in order
(omit function.cc if you didn't write one):

```cpp:function.hh
...
```

```cpp:function.cc
... only if you wrote one ...
```

```cpp:test.cc
...
```
"""


def _build_task(
    cm: ClassModel,
    method_qname: str,
    gemmi_hh: str,
    gemmi_cc: str | None,
    gemmi_test: str,
) -> str:
    method_name = method_qname.rsplit("::", 1)[-1]
    parts: list[str] = []
    parts.append("## Task")
    parts.append(
        f"Lift the verified gemmi port of `{method_qname}` into a method "
        f"`{method_name}` of the class **`{cm.qualified_name}`**, and adapt its "
        f"test to construct the class. Move only — do not change the logic."
    )
    parts.append(cm.contract_prompt())
    parts.append("## Verified gemmi port — function.hh (this currently compiles & passes)")
    parts.append(f"```cpp\n{gemmi_hh.rstrip()}\n```")
    if gemmi_cc and gemmi_cc.strip():
        parts.append("## Verified gemmi port — function.cc")
        parts.append(f"```cpp\n{gemmi_cc.rstrip()}\n```")
    parts.append("## Verified gemmi test — adapt this to construct the class")
    parts.append("_Keep every assertion and expected value identical._")
    parts.append(f"```cpp\n{gemmi_test.rstrip()}\n```")
    return "\n\n".join(parts)


def run_lift_agent(
    conn: sqlite3.Connection,
    *,
    class_model: ClassModel,
    method_qname: str,
    gemmi_dir: Path,
    work_dir: Path,
    dep_includes: list[Path],
    dep_sources: list[Path],
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> tuple[dict[str, str] | None, str]:
    """Drive the lift. Returns ({filename: contents}, trace) or (None, trace)."""
    gemmi_hh = (gemmi_dir / "function.hh").read_text()
    gemmi_cc_path = gemmi_dir / "function.cc"
    gemmi_cc = gemmi_cc_path.read_text() if gemmi_cc_path.exists() else None
    gemmi_test = (gemmi_dir / "test.cc").read_text()

    work_dir.mkdir(parents=True, exist_ok=True)
    compile_handler, run_handler, get_errors_handler, write_file_handler, \
        read_file_handler, patch_file_handler, compiled_ok = make_tool_handlers(
            work_dir, dep_includes, dep_sources,
            original_test_cc=gemmi_test, conn=conn,
        )

    def dispatch(name: str, args: dict) -> str:
        if name == "compile_gemmi":
            return compile_handler(
                args.get("function_hh", ""), args.get("test_cc", ""),
                args.get("function_cc") or None,
            )
        if name == "write_gemmi_file":
            return write_file_handler(args.get("filename", ""), args.get("contents", ""))
        if name == "read_gemmi_file":
            return read_file_handler(args.get("filename", ""))
        if name == "patch_gemmi_file":
            old_s, new_s = args.get("old_string"), args.get("new_string")
            if old_s is None or new_s is None:
                return "Error: patch_gemmi_file requires 'filename', 'old_string', 'new_string'."
            return patch_file_handler(args.get("filename", ""), old_s, new_s,
                                      bool(args.get("replace_all", False)))
        if name == "run_gemmi_test":
            return run_handler()
        if name == "get_compile_errors":
            return get_errors_handler()
        return f"Unknown tool: {name}"

    task = _build_task(class_model, method_qname, gemmi_hh, gemmi_cc, gemmi_test)
    messages = [
        {"role": "system", "content": LIFT_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    (work_dir / "prompt.txt").write_text(
        f"=== SYSTEM ===\n{LIFT_SYSTEM_PROMPT}\n\n=== USER ===\n{task}\n"
    )
    trace = _TraceWriter(work_dir / "agent_trace.txt")
    trace.append("=== LIFT AGENT TRACE ===\n")
    trace.append(f"[user]\n{textwrap.indent(task, '  ')}\n")

    def _run_tool_calls(tool_calls: list[dict]) -> list[dict]:
        results = []
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result = dispatch(name, args)
            short = "{...}" if name in ("compile_gemmi", "write_gemmi_file") else json.dumps(args)
            trace.append(f"  → {name}({short})")
            trace.append(textwrap.indent(result, "      ") + "\n")
            trace.append_call(f"{name}({short})")
            results.append({"role": "tool", "content": result})
        return results

    final_blocks: dict[str, str] | None = None
    unverified_nudged = False
    for turn in range(MAX_TURNS):
        print(f"  [lift] turn {turn + 1}/{MAX_TURNS} ...", end="", flush=True)
        data = _chat(messages, model, LIFT_TOOLS)
        _log_llm_timing(data, stage="lift", turn=turn + 1, verbose=verbose, trace_lines=trace)
        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls") or []
        thinking = msg.get("thinking", "") or ""
        content = msg.get("content", "") or ""
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        if thinking:
            trace.append(f"[thinking — turn {turn + 1}]\n{textwrap.indent(thinking, '  ')}\n")

        # Escape hatch: the agent may declare the provided port itself broken
        # (defense-in-depth — the source-port gate normally catches this first).
        if "INPUT_BROKEN" in content.upper():
            print(" done (INPUT_BROKEN)", flush=True)
            trace.append("[lift] Agent reported INPUT_BROKEN — the provided port "
                         "does not compile as given.\n")
            final_blocks = None
            break

        if not tool_calls:
            # Don't accept a final answer whose latest compile is red — the model
            # must reconcile with compiler feedback first. Nudge once, then yield.
            if not compiled_ok() and not unverified_nudged:
                unverified_nudged = True
                nudge = (
                    "Your latest code has NOT compiled+passed yet, so do not give a "
                    "final answer. Write the corrected file(s) with write_gemmi_file "
                    "(writing test.cc auto-compiles and runs). Remember: the only "
                    "allowed edits are the first-parameter→member substitution and "
                    "the name/signature change — copy the rest of the body verbatim. "
                    "If the PROVIDED port fails to compile on lines you did NOT "
                    "change, reply with exactly INPUT_BROKEN and nothing else."
                )
                messages.append({"role": "user", "content": nudge})
                trace.append(f"[lift] unverified-final nudge — turn {turn + 1}\n")
                print(" (unverified — nudged)", flush=True)
                continue
            print(" done (final answer)", flush=True)
            trace.append(f"[assistant — final]\n{textwrap.indent(content, '  ')}\n")
            final_blocks = extract_blocks(content) or None
            break
        names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tool_calls)
        print(f" → {names}", flush=True)
        trace.append(f"[assistant — turn {turn + 1}, {len(tool_calls)} tool call(s)]")
        messages.extend(_run_tool_calls(tool_calls))
    else:
        trace.append("[lift] Turn limit reached.\n")

    text = trace.text()
    trace.close()
    return final_blocks, text
