"""Combined agentic port: function.hh (+ optional function.cc) + test.cc in one session.

The original MMDB function source and its MMDB-based test are both supplied.
The agent produces a gemmi equivalent of the function AND a gemmi version of
the test that exercises it — compiled and linked as a single unit so
signatures agree by construction.

Frozen: every EXPECT_* / ASSERT_* line from the original test.
"""
from __future__ import annotations

import json
import re
import sqlite3
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

from ..oracle.agent import (
    OLLAMA_CHAT_URL, TOOLS, _dispatch,
    _EXTENSION_TURNS, _MAX_EXTENSIONS, _EXTENSION_PROMPT,
    _tool_resolve_includes, _has_unresolved_includes,
    _tool_grep_codebase,
    _TraceWriter,
    _chat,
    _is_degenerate_thinking,
    NUDGE_EVERY_N_TURNS,
    NO_COMPILE_AFTER,
)

# Format-reminder nudge (injected every NUDGE_EVERY_N_TURNS turns) — keeps
# the output-format spec near the end of the context where attention is
# strongest. The gemmi port has the strictest format requirement of the
# three agents (two or three labelled fenced blocks), so a tighter reminder
# pays for itself.
_GEMMI_NUDGE = (
    "Reminder: when you stop calling tools, your final reply must be exactly "
    "two or three fenced code blocks, labelled:\n"
    "  ```cpp:function.hh\n  ...\n  ```\n"
    "  ```cpp:function.cc          (optional — only if needed)\n  ...\n  ```\n"
    "  ```cpp:test.cc\n  ...\n  ```\n"
    "Do not summarise. Do not narrate. If you have a working draft, call "
    "compile_gemmi NOW to validate it before finalising."
)

_GEMMI_NO_COMPILE_NUDGE = (
    "WARNING: you have not attempted compile_gemmi yet. "
    "Stop researching and DRAFT your best function.hh + test.cc (and "
    "optionally function.cc) NOW, then call compile_gemmi. The compiler's "
    "error messages are far more useful than further speculation about "
    "gemmi APIs. Failures are expected — you have multiple retries to fix "
    "them. Action over analysis."
)
from ..oracle.compile import GEMMI_INCLUDE
from ..oracle.notes import load_notes, render_notes_for_prompt
from .compile import MAX_COMPILE_ATTEMPTS, compile_gemmi, run_gemmi_test_binary

# Absolute paths to data files (pdb/cif/mtz/map/ent) inside the original test
# source. These fixtures are validated by the oracle stage and MUST carry over
# to the gemmi test verbatim — surfacing them in the prompt prevents the agent
# from spending tool calls on inspect_pdb / grep_codebase to "verify" them.
_FIXTURE_PATH_RE = re.compile(r'"(/[^"\s]+\.(?:pdb|cif|mmcif|ent|mtz|map))"')


def _extract_test_fixtures(test_cc: str) -> list[str]:
    seen: list[str] = []
    for m in _FIXTURE_PATH_RE.finditer(test_cc):
        path = m.group(1)
        if path not in seen:
            seen.append(path)
    return seen

GEMMI_CHEAT_SHEET = """\
## gemmi quick reference (verified against the installed headers)

Headers you will almost certainly need:
  #include <gemmi/model.hpp>      // Structure, Model, Chain, Residue, Atom, CRA
  #include <gemmi/pdb.hpp>        // read_pdb_file(path)
  #include <gemmi/mmread.hpp>     // read_structure(path)  — auto-detects format
  #include <gemmi/neighbor.hpp>   // NeighborSearch
  #include <gemmi/contact.hpp>    // ContactSearch, for_each_contact
  #include <gemmi/math.hpp>       // Vec3, Position (Position is a Vec3 of doubles)
  #include <gemmi/unitcell.hpp>   // UnitCell, Fractional, Position helpers

Loading a PDB — this is the only idiom that works:
  gemmi::Structure st = gemmi::read_pdb_file("/abs/path/to/file.pdb");
  st.setup_entities();  // populate entity_type on residues (required by some APIs)

Traversal — every level is a std::vector, so you iterate, not GetXxx():
  for (gemmi::Model&   model   : st.models)
  for (gemmi::Chain&   chain   : model.chains)
  for (gemmi::Residue& residue : chain.residues)
  for (gemmi::Atom&    atom    : residue.atoms) { ... }

Key MMDB → gemmi accessor map (use these VERBATIM — do not invent variants):
  mol->GetModel(1)                → st.models[0]            // 0-indexed!
  chain->GetChainID()             → chain.name              // field, not method
  residue->GetResName()           → residue.name            // field
  residue->GetSeqNum()            → residue.seqid.num.value // .seqid is ResidueId base
  residue->GetInsCode()           → residue.seqid.icode     // char, not const char*
  atom->GetAtomName()             → atom.name               // std::string field
  atom->GetElementName()          → atom.element.name()
  atom->x, atom->y, atom->z       → atom.pos.x, atom.pos.y, atom.pos.z
  atom->occupancy                 → atom.occ
  atom->tempFactor                → atom.b_iso
  chain->GetNumberOfResidues()    → chain.residues.size()
  residue->GetNumberOfAtoms()     → residue.atoms.size()

No gemmi equivalents exist for MMDB selection / Manager.Select / NewSelection —
iterate manually, or use gemmi::NeighborSearch for distance queries.

NeighborSearch — distance queries against a Model:
  gemmi::NeighborSearch ns(st.models[0], st.cell, /*max_radius=*/5.0);
  ns.populate(/*include_h=*/false);   // MUST call before any find_*
  std::vector<gemmi::NeighborSearch::Mark*> hits =
      ns.find_atoms(atom.pos, /*alt=*/'\\0', /*min_dist=*/0.0, /*radius=*/4.0);
  for (auto* m : hits) {
      gemmi::CRA cra = m->to_cra(st.models[0]);   // gives chain/residue/atom
      // cra.chain, cra.residue, cra.atom are POINTERS (may be nullptr)
  }

ContactSearch — pairs of atoms within a radius:
  gemmi::ContactSearch cs(/*search_radius=*/4.0);
  cs.ignore = gemmi::ContactSearch::Ignore::SameResidue;  // or AdjacentResidues, SameChain, SameAsu, Nothing
  cs.setup_atomic_radii(1.0, 0.0);                        // optional, for VdW-aware filtering
  std::vector<gemmi::ContactSearch::Result> contacts = cs.find_contacts(ns);
  for (const auto& c : contacts) {
      // c.partner1, c.partner2 are CRA; c.dist_sq is double; c.image_idx is int
  }

CRA shape (gemmi/model.hpp):
  struct CRA { Chain* chain; Residue* residue; Atom* atom; };  // ALL POINTERS

There is NO top-level <gemmi.hpp>. There is NO atom.get_pos() or st.n_atoms().
Vec3 operator* is component-wise; for dot product use v.dot(w); for squared
length use v.length_sq().

Everything above is in code_graph.db — lookup_type("gemmi::NeighborSearch"),
list_methods("gemmi::ContactSearch"), find_symbol("read_pdb_file"), etc. will
show the real API. Some signatures involving std::string / std::vector may
appear as "int" in the DB summary due to a libclang template-resolution quirk;
when in doubt, read the header directly from /opt/homebrew/include/gemmi/
with read_file or grep_codebase (the gemmi tree is searched alongside coot).
"""

GEMMI_SYSTEM_PROMPT = f"""\
You are porting ONE C++ function from the MMDB API to the gemmi API AND
translating its Google Test, in the same session.

{GEMMI_CHEAT_SHEET}

## Artifacts to produce

  A. function.hh — header with declaration and #include <gemmi/...> deps.
     Use `#pragma once`. If the body is short, put it here as `inline`.
  B. function.cc — OPTIONAL. Only emit if the body is long or uses
     translation-unit-private helpers. Otherwise omit it entirely.
  C. test.cc — the gemmi-translated Google Test, #include "function.hh".

## Rules

1. Preserve every EXPECT_* / ASSERT_* line's semantic fact — same compared
   value, same comparison operator. You MAY rewrite the left-hand-side
   accessor when the type changes (e.g. `res->GetSeqNum()` becomes
   `res->seqid.num.value`), but you MAY NOT change the expected value or
   relax the check. The original expected numbers are the correctness
   oracle.
2. Port the function semantics 1:1 — same output for the same input.
3. **Naming**: keep the original function's C++ namespace exactly as-is and
   append `_gemmi` to the function name. For example, if the original is
   `coot::angle(...)` the ported function MUST be declared and defined as
   `coot::angle_gemmi(...)`. Do NOT wrap it in a `gemmi::` namespace or any
   other namespace. The task below states the exact target name — use it
   verbatim.
4. The function signature must match what test.cc calls. Design them together.
5. Use the DB tools (lookup_type, list_methods, find_header, find_symbol)
   BEFORE writing any gemmi name. When lookup_type reports an ambiguous
   name, retry with the fully-qualified form. Do not invent APIs.
6. grep_codebase searches both coot and the gemmi header tree — use it
   when you need to see a usage pattern.
7. Link target: test.cc (+ function.cc if present) against -lgemmi_cpp and
   -lgtest. No MMDB, no clipper, no coot libraries.
8. Call compile_gemmi with your drafts. Fix errors until it builds. Then
   call run_gemmi_test to confirm assertions pass. Max {MAX_COMPILE_ATTEMPTS} compile attempts.

Final output format (ONE response, THREE fenced blocks in this exact order):

```cpp:function.hh
... header contents ...
```

```cpp:function.cc
... only if needed; otherwise omit this block entirely ...
```

```cpp:test.cc
... test contents ...
```

If you omit function.cc, just skip the middle block.\
"""

_COMPILE_TOOL = {
    "type": "function",
    "function": {
        "name": "compile_gemmi",
        "description": (
            "Write the supplied sources to disk (function.hh, test.cc, and "
            "optionally function.cc) and compile them as one unit linked "
            f"against -lgemmi_cpp and -lgtest. Max {MAX_COMPILE_ATTEMPTS} attempts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "function_hh": {"type": "string",
                                "description": "Contents of function.hh"},
                "test_cc":     {"type": "string",
                                "description": "Contents of test.cc"},
                "function_cc": {"type": "string",
                                "description": "Optional contents of function.cc "
                                               "(omit for header-only)"},
            },
            "required": ["function_hh", "test_cc"],
        },
    },
}

_RUN_TOOL = {
    "type": "function",
    "function": {
        "name": "run_gemmi_test",
        "description": "Run the last compiled test binary and return GoogleTest output.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_GET_ERRORS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_compile_errors",
        "description": "Return the full last compile log without truncation.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

GEMMI_TOOLS = TOOLS + [_COMPILE_TOOL, _RUN_TOOL, _GET_ERRORS_TOOL]


def _make_tool_handlers(gemmi_subdir: Path) -> tuple[callable, callable, callable]:
    attempts       = [0]
    last_binary    = [None]
    last_error_log = [None]

    def compile_handler(function_hh: str, test_cc: str,
                        function_cc: str | None = None) -> str:
        if attempts[0] >= MAX_COMPILE_ATTEMPTS:
            return (f"Compile limit reached ({MAX_COMPILE_ATTEMPTS}). "
                    "Output your best drafts as the final fenced blocks.")

        # Pre-flight include check across all three files — free fix cycle.
        sections: list[str] = []
        for label, body in (("function.hh", function_hh),
                            ("function.cc", function_cc),
                            ("test.cc",     test_cc)):
            if not body:
                continue
            report = _tool_resolve_includes(body)
            if _has_unresolved_includes(report):
                sections.append(f"--- {label} ---\n{report}")
        if sections:
            return (
                "Include check FAILED (this does not count against your "
                f"{MAX_COMPILE_ATTEMPTS} compile attempts). Fix the paths "
                "below and call compile_gemmi again:\n"
                + "\n\n".join(sections)
            )

        attempts[0] += 1
        gemmi_subdir.mkdir(exist_ok=True)

        hh_path   = gemmi_subdir / "function.hh"
        test_path = gemmi_subdir / "test.cc"
        cc_path   = gemmi_subdir / "function.cc"
        hh_path.write_text(function_hh)
        test_path.write_text(test_cc)
        if function_cc:
            cc_path.write_text(function_cc)
            fn_cc_arg = cc_path
        else:
            if cc_path.exists():
                cc_path.unlink()
            fn_cc_arg = None

        test_bin = gemmi_subdir / "test_check"
        success, output = compile_gemmi(test_path, test_bin, fn_cc_arg)

        error_log = gemmi_subdir / "compile_error.log"
        error_log.write_text(output)
        lines = output.splitlines()
        if len(lines) > 120:
            output = ("\n".join(lines[:120])
                      + f"\n... ({len(lines) - 120} more — use get_compile_errors)")
        if success:
            last_binary[0] = test_bin
            return f"Compilation succeeded (attempt {attempts[0]}/{MAX_COMPILE_ATTEMPTS})."
        last_binary[0] = None
        last_error_log[0] = error_log
        return f"Compilation FAILED (attempt {attempts[0]}/{MAX_COMPILE_ATTEMPTS}):\n{output}"

    def run_handler() -> str:
        if last_binary[0] is None:
            return "No compiled binary — call compile_gemmi first."
        success, output = run_gemmi_test_binary(last_binary[0])
        lines = output.splitlines()
        if len(lines) > 100:
            output = "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines)"
        status = "All tests PASSED." if success else "Some tests FAILED."
        return f"{status}\n{output}"

    def get_errors_handler() -> str:
        if last_error_log[0] is None or not last_error_log[0].exists():
            return "No compile error log available."
        return last_error_log[0].read_text()

    return compile_handler, run_handler, get_errors_handler


_BLOCK_RE = re.compile(
    r"```(?:cpp|c\+\+)?(?::([^\n]+))?\n(.*?)```",
    re.DOTALL,
)


def _extract_blocks(content: str) -> dict[str, str]:
    """Pull named fenced blocks out of the assistant's final message.

    Accepts labelled fences like ```cpp:function.hh or falls back to ordering
    (hh, cc, test) if labels are missing.
    """
    found: dict[str, str] = {}
    unlabelled: list[str] = []
    for label, body in _BLOCK_RE.findall(content):
        body = body.strip()
        label = (label or "").strip().lower()
        if "function.hh" in label or label.endswith(".hh"):
            found["function.hh"] = body
        elif "function.cc" in label or label.endswith("function.cc"):
            found["function.cc"] = body
        elif "test.cc" in label or label.endswith("test.cc"):
            found["test.cc"] = body
        else:
            unlabelled.append(body)
    if unlabelled:
        keys = ["function.hh", "function.cc", "test.cc"]
        for key in keys:
            if key not in found and unlabelled:
                found[key] = unlabelled.pop(0)
    return found


def generate_gemmi_port_with_agent(
    conn: sqlite3.Connection,
    original_function_src: str,
    function_qname: str,
    original_test_cc: str,
    gemmi_subdir: Path,
    model: str,
    verbose: bool = False,
) -> tuple[dict[str, str] | None, str]:
    """Return ({file_name: contents, ...}, trace_text) or (None, trace) on failure."""
    compile_handler, run_handler, get_errors_handler = _make_tool_handlers(gemmi_subdir)

    def dispatch(name: str, args: dict) -> str:
        if name == "compile_gemmi":
            return compile_handler(
                args.get("function_hh", ""),
                args.get("test_cc", ""),
                args.get("function_cc") or None,
            )
        if name == "run_gemmi_test":
            return run_handler()
        if name == "get_compile_errors":
            return get_errors_handler()
        # Widen grep to include the gemmi header tree — most API discovery
        # during a port needs to see gemmi usage, which isn't in PROJECT_ROOT.
        if name == "grep_codebase":
            return _tool_grep_codebase(
                args["pattern"],
                args.get("glob"),
                extra_roots=[GEMMI_INCLUDE],
            )
        return _dispatch(conn, name, args)

    parts: list[str] = []

    # Derive the target name: same namespace, function base name + _gemmi suffix.
    # e.g. "coot::molecule_t::angle" → target "coot::molecule_t::angle_gemmi"
    _ns_parts = function_qname.rsplit("::", 1)
    if len(_ns_parts) == 2:
        _target_name = f"{_ns_parts[0]}::{_ns_parts[1]}_gemmi"
    else:
        _target_name = f"{function_qname}_gemmi"

    parts.append("## Task")
    parts.append(
        f"Port `{function_qname}` to gemmi AND translate its MMDB test in one "
        f"pass. The ported function MUST be named **`{_target_name}`** — same "
        "namespace as the original, with `_gemmi` appended to the function "
        "name. Do NOT place it inside a `gemmi::` namespace. "
        "Design the function signature and the test's call site together. "
        "Use the tools to resolve gemmi types. Compile and run before finalising."
    )

    fixtures = _extract_test_fixtures(original_test_cc)
    if fixtures:
        parts.append(
            "## Test fixtures (use these paths VERBATIM in the gemmi test — "
            "do NOT call inspect_pdb or grep_codebase to verify them)"
        )
        parts.append("\n".join(f"  - {p}" for p in fixtures))

    parts.append("## Original MMDB function")
    parts.append(f"```cpp\n{original_function_src.rstrip()}\n```")

    parts.append("## Original MMDB test")
    parts.append("_FREEZE every `EXPECT_*` — keep the assertions identical._")
    parts.append(f"```cpp\n{original_test_cc.rstrip()}\n```")

    notes = load_notes(gemmi_subdir.parent / "oracle" / "notes.json")
    if notes:
        rendered = render_notes_for_prompt(notes, audience="gemmi")
        if rendered:
            parts.append("## Validated facts from oracle stage")
            parts.append(
                "_Carry these over where they still apply; treat port caveats "
                "as concrete design hints._"
            )
            parts.append(f"```\n{rendered.rstrip()}\n```")

    user_content = "\n\n".join(parts)

    messages: list[dict] = [
        {"role": "system", "content": GEMMI_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    gemmi_subdir.mkdir(parents=True, exist_ok=True)
    (gemmi_subdir / "prompt.txt").write_text(
        f"=== SYSTEM ===\n{GEMMI_SYSTEM_PROMPT}\n\n"
        f"=== USER ===\n{user_content}\n"
    )

    trace_lines = _TraceWriter(gemmi_subdir / "agent_trace.txt")
    trace_lines.append("=== GEMMI COMBINED AGENT TRACE ===\n")
    trace_lines.append(f"[user]\n{textwrap.indent(user_content, '  ')}\n")

    final_blocks: dict[str, str] | None = None
    last_draft: list[dict[str, str] | None] = [None]
    call_counts: dict[str, int] = {}
    tool_cache: dict[str, str] = {}
    REPEAT_LIMIT = 3
    NO_CACHE = {"compile_gemmi", "run_gemmi_test", "get_compile_errors", "leave_note"}
    no_compile_warned = [False]

    def _save_draft_from_compile(args: dict) -> None:
        hh = args.get("function_hh") or ""
        tc = args.get("test_cc") or ""
        if len(hh) > 50 and len(tc) > 100 and "#include" in tc:
            last_draft[0] = {
                "function.hh": hh,
                "test.cc": tc,
                **({"function.cc": args["function_cc"]}
                   if args.get("function_cc") else {}),
            }

    def _run_tool_calls(tool_calls: list[dict]) -> list[dict]:
        results: list[dict] = []
        for call in tool_calls:
            fn_info = call.get("function", {})
            name    = fn_info.get("name", "")
            args    = fn_info.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name == "compile_gemmi":
                _save_draft_from_compile(args)
            hash_args = {k: v for k, v in args.items()
                         if k not in ("function_hh", "function_cc", "test_cc")}
            key = f"{name}:{json.dumps(hash_args, sort_keys=True)}"
            call_counts[key] = call_counts.get(key, 0) + 1
            if name not in NO_CACHE and key in tool_cache:
                cached = tool_cache[key]
                note = (
                    "(cached — you already called this with the same arguments. "
                    "Use the answer below; do not re-query.)\n"
                )
                trace_lines.append(f"  → [cached × {call_counts[key]}] {name}({json.dumps(hash_args)})")
                results.append({"role": "tool", "content": note + cached})
                continue
            if call_counts[key] > REPEAT_LIMIT and name not in ("compile_gemmi", "run_gemmi_test"):
                nudge = (
                    f"You have called {name} with these arguments {call_counts[key]} times. "
                    "Stop repeating — proceed to compile_gemmi with your best drafts."
                )
                trace_lines.append(f"  → {name}(repeated — nudged)")
                results.append({"role": "tool", "content": nudge})
                continue
            if verbose:
                display = ({"function_hh": "...", "test_cc": "...",
                            "function_cc": "..." if args.get("function_cc") else None}
                           if name == "compile_gemmi" else args)
                print(f"  tool: {name}({display})")
            result_text = dispatch(name, args)
            result_lines = result_text.splitlines()
            if len(result_lines) > 150:
                result_text = ("\n".join(result_lines[:150])
                               + f"\n... ({len(result_lines) - 150} more lines)")
            trace_lines.append(
                f"  → {name}({json.dumps(args) if name != 'compile_gemmi' else '{...}'})"
            )
            trace_lines.append(textwrap.indent(result_text, "      ") + "\n")
            if name not in NO_CACHE:
                tool_cache[key] = result_text
            results.append({"role": "tool", "content": result_text})
        return results

    def _is_usable(blocks: dict[str, str] | None) -> bool:
        if not blocks:
            return False
        return ("function.hh" in blocks and "test.cc" in blocks
                and "#include" in blocks["test.cc"])

    for turn in range(25):
        data = _chat(messages, model, GEMMI_TOOLS)
        msg  = data.get("message", {})
        tool_calls        = msg.get("tool_calls") or []
        thinking          = msg.get("thinking",  "") or ""
        assistant_content = msg.get("content",   "") or ""
        messages.append({"role": "assistant", "content": assistant_content,
                         "tool_calls": tool_calls})
        if thinking:
            trace_lines.append(f"[thinking — turn {turn + 1}]\n{textwrap.indent(thinking, '  ')}\n")

        # Degenerate-thinking guard: if this turn's thinking is pathologically
        # repetitive, the model has likely saturated num_ctx with junk and the
        # rest of the response is unrecoverable. Break out of the loop
        # immediately and let the rescue prompt have a clean context window.
        degen, diag = _is_degenerate_thinking(thinking)
        if degen:
            trace_lines.append(
                f"[agent] {diag} — aborting loop, will issue rescue.\n"
            )
            break

        if not tool_calls:
            trace_lines.append(f"[assistant — final]\n{textwrap.indent(assistant_content, '  ')}\n")
            final_blocks = _extract_blocks(assistant_content) or None
            break
        trace_lines.append(f"[assistant — turn {turn + 1}, {len(tool_calls)} tool call(s)]")
        messages.extend(_run_tool_calls(tool_calls))

        if (NO_COMPILE_AFTER and not no_compile_warned[0]
                and (turn + 1) >= NO_COMPILE_AFTER
                and not any(k.startswith("compile_gemmi:") for k in call_counts)):
            messages.append({"role": "user", "content": _GEMMI_NO_COMPILE_NUDGE})
            trace_lines.append(f"[no-compile nudge — turn {turn + 1}]\n{textwrap.indent(_GEMMI_NO_COMPILE_NUDGE, '  ')}\n")
            no_compile_warned[0] = True

        if NUDGE_EVERY_N_TURNS and (turn + 1) % NUDGE_EVERY_N_TURNS == 0:
            messages.append({"role": "user", "content": _GEMMI_NUDGE})
            trace_lines.append(f"[nudge — turn {turn + 1}]\n{textwrap.indent(_GEMMI_NUDGE, '  ')}\n")
    else:
        trace_lines.append("[agent] Turn limit reached.\n")

    if not _is_usable(final_blocks) and _is_usable(last_draft[0]):
        trace_lines.append("[agent] Falling back to last compile_gemmi draft.\n")
        final_blocks = last_draft[0]
    elif not _is_usable(final_blocks):
        trace_lines.append("[agent] No usable output — issuing rescue prompt.\n")
        messages.append({"role": "user", "content": (
            "STOP. Do not call any tools. Output your best attempt NOW as "
            "three fenced blocks labelled ```cpp:function.hh, optionally "
            "```cpp:function.cc, and ```cpp:test.cc."
        )})
        try:
            data = _chat(messages, model, tools=[])
            assistant_content = (data.get("message") or {}).get("content") or ""
            trace_lines.append(f"[assistant — rescue]\n{textwrap.indent(assistant_content, '  ')}\n")
            rescued = _extract_blocks(assistant_content)
            if _is_usable(rescued):
                final_blocks = rescued
            elif _is_usable(last_draft[0]):
                final_blocks = last_draft[0]
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            trace_lines.append(f"[agent] Rescue call failed: {e}\n")
            if _is_usable(last_draft[0]):
                final_blocks = last_draft[0]

    text = trace_lines.text()
    trace_lines.close()
    return final_blocks, text
