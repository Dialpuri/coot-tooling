"""Agentic Google Test generation — model calls tools to resolve headers,
look up types, and iteratively compile its draft before finalising."""
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
)
from ..oracle.runner.results import OracleResult
from .compile import MAX_COMPILE_ATTEMPTS, compile_test_cc, run_test_binary

TEST_SYSTEM_PROMPT = """\
You are converting a C++ oracle program into a Google Test suite (test.cc).

Rules:
1. Keep all setup code (loading PDB/MTZ, constructing objects, calling the
   function) identical to the oracle.
2. For each test case provided, write assertions using the observed values.
   Choose the assertion type carefully:
   - Exact integers or booleans: EXPECT_EQ / EXPECT_TRUE / EXPECT_FALSE
   - Floating-point values: EXPECT_FLOAT_EQ or EXPECT_NEAR, never EXPECT_EQ
   - Large strings (e.g. PDB file contents): EXPECT_FALSE(s.empty()), size range
     checks, and EXPECT_NE(s.find("keyword"), std::string::npos) — do NOT
     hardcode exact byte counts that will break on minor formatting changes
   - Null/non-null pointers: EXPECT_NE(ptr, nullptr) or EXPECT_EQ(ptr, nullptr)
   - If the function returns void, assert observable side effects or at minimum
     assert the function does not crash (the test passes by reaching the end)
3. If the oracle has multiple test cases, wrap them all in a SINGLE
   TEST(OracleTest, <FunctionName>) block. Use a nested scope { ... } or a
   comment to label each case.
4. Add a main() that calls RUN_ALL_TESTS().
5. Remove all INPUT/OUTPUT std::cout lines — assertions only.

Mandatory steps before outputting the final program:
6. Call resolve_includes on your draft FIRST to verify every #include "..."
   header resolves correctly. Fix any that do not.
7. Call compile_test with your draft. Fix all compiler errors and call again
   until it succeeds (max 3 attempts). If compilation fails with a long error
   log, call get_compile_errors to see the full output.
8. Once it compiles, call run_test to check for assertion failures. Fix any
   mismatches and recompile.
9. Output only the final, compiling and passing C++ source in a single ```cpp
   block.\
"""

_RUN_TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "run_test",
        "description": (
            "Run the last successfully compiled test binary and return the "
            "GoogleTest output. Use this after compile_test succeeds to check "
            "for failing assertions. Fix any EXPECT_EQ mismatches and recompile."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_COMPILE_TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "compile_test",
        "description": (
            "Write the supplied C++ code as test.cc and attempt to compile it. "
            "Returns compiler output. Fix any errors shown and call again. "
            f"Maximum {MAX_COMPILE_ATTEMPTS} attempts — stop iterating if the "
            "limit is reached and output whatever compiles."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The complete C++ test source to compile",
                },
            },
            "required": ["code"],
        },
    },
}

_GET_COMPILE_ERRORS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_compile_errors",
        "description": (
            "Return the full compiler output from the last compile_test call, "
            "without any line truncation. Use this when compile_test showed "
            "'... N more lines truncated'."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

TEST_TOOLS = TOOLS + [_COMPILE_TEST_TOOL, _RUN_TEST_TOOL, _GET_COMPILE_ERRORS_TOOL]


def _make_tool_handlers(test_subdir: Path) -> tuple[callable, callable, callable]:
    """Return (compile_handler, run_handler, get_errors_handler) sharing state about the last build."""
    attempts       = [0]
    last_binary    = [None]   # Path | None — set when a compile succeeds
    last_error_log = [None]   # Path | None

    def compile_handler(code: str) -> str:
        if attempts[0] >= MAX_COMPILE_ATTEMPTS:
            return (
                f"Compile limit reached ({MAX_COMPILE_ATTEMPTS} attempts). "
                "Output your best draft as the final ```cpp block."
            )
        attempts[0] += 1

        test_subdir.mkdir(exist_ok=True)
        test_cc  = test_subdir / "test.cc"
        test_bin = test_subdir / "test_check"
        test_cc.write_text(code)

        success, output = compile_test_cc(test_cc, test_bin)

        # Always save full log so the agent can read it if needed.
        error_log = test_subdir / "compile_error.log"
        error_log.write_text(output)

        lines = output.splitlines()
        if len(lines) > 100:
            truncated = "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines truncated)"
            truncated += f"\nFull log saved to: {error_log} — use get_compile_errors to see more."
            output = truncated

        if success:
            last_binary[0] = test_bin
            return f"Compilation succeeded (attempt {attempts[0]}/{MAX_COMPILE_ATTEMPTS})."
        last_binary[0] = None
        last_error_log[0] = error_log
        return f"Compilation FAILED (attempt {attempts[0]}/{MAX_COMPILE_ATTEMPTS}):\n{output}"

    def run_handler() -> str:
        if last_binary[0] is None:
            return "No compiled binary available — call compile_test first."
        success, output = run_test_binary(last_binary[0])
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


def _chat(messages: list[dict], model: str, tools: list[dict]) -> dict:
    payload = json.dumps({
        "model":    model,
        "messages": messages,
        "tools":    tools,
        "stream":   False,
        "think":    True,
    }).encode()
    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def generate_test_with_agent(
    conn: sqlite3.Connection,
    oracle_cc_text: str,
    oracle_result: OracleResult,
    test_subdir: Path,
    model: str,
    oracle_trace: str | None = None,
    verbose: bool = False,
) -> tuple[str | None, str]:
    """Run the agentic test generation loop.

    Returns (test_code, trace_text).
    test_code is None if the model produced nothing usable.
    """
    compile_handler, run_handler, get_errors_handler = _make_tool_handlers(test_subdir)

    def dispatch(name: str, args: dict) -> str:
        if name == "compile_test":
            return compile_handler(args["code"])
        if name == "run_test":
            return run_handler()
        if name == "get_compile_errors":
            return get_errors_handler()
        return _dispatch(conn, name, args)

    # Format structured cases
    if oracle_result.cases:
        cases_text = f"Oracle produced {len(oracle_result.cases)} test case(s):\n"
        for i, case in enumerate(oracle_result.cases, 1):
            cases_text += f"\nCase {i}:\n"
            for k, v in case["inputs"].items():
                cases_text += f"  INPUT  {k}: {v}\n"
            for k, v in case["outputs"].items():
                cases_text += f"  OUTPUT {k}: {v}\n"
    else:
        cases_text = f"Observed output when run:\n{oracle_result.stdout}\n"
        cases_text += "There may be warnings at the top — the INPUT/OUTPUT lines are the ground truth.\n"

    # Compact oracle trace summary (tool calls only)
    trace_summary = ""
    if oracle_trace:
        tool_lines = [l.strip() for l in oracle_trace.splitlines() if l.strip().startswith("→")]
        if tool_lines:
            trace_summary = (
                "\nOracle agent confirmed these lookups during generation "
                "(types/headers already verified):\n"
                + "\n".join(f"  {l}" for l in tool_lines[:20])
            )
            if len(tool_lines) > 20:
                trace_summary += f"\n  ... ({len(tool_lines) - 20} more)"

    user_content = (
        f"Here is the oracle program:\n\n```cpp\n{oracle_cc_text}\n```\n\n"
        + cases_text
        + trace_summary
        + "\n\nConvert this into a Google Test suite. Use the tools to verify "
        "headers, look up any types you are unsure about, then compile and run "
        "before finalising."
    )

    messages: list[dict] = [
        {"role": "system", "content": TEST_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    test_subdir.mkdir(parents=True, exist_ok=True)
    (test_subdir / "prompt.txt").write_text(
        f"=== SYSTEM ===\n{TEST_SYSTEM_PROMPT}\n\n"
        f"=== USER ===\n{user_content}\n"
    )

    trace_lines: list[str] = [
        "=== TEST AGENT TRACE ===\n",
        f"[user]\n{textwrap.indent(user_content, '  ')}\n",
    ]

    test_code: str | None = None
    last_draft: list[str | None] = [None]
    call_counts: dict[str, int] = {}
    REPEAT_LIMIT = 3

    def _save_draft(code: str) -> None:
        if code and len(code) > 100 and "#include" in code:
            last_draft[0] = code

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
            if name == "compile_test" and isinstance(args.get("code"), str):
                _save_draft(args["code"])
            hash_args = {k: v for k, v in args.items() if k != "code"}
            key = f"{name}:{json.dumps(hash_args, sort_keys=True)}"
            call_counts[key] = call_counts.get(key, 0) + 1
            if call_counts[key] > REPEAT_LIMIT and name not in ("compile_test", "run_test"):
                nudge = (
                    f"You have called {name} with these arguments {call_counts[key]} times. "
                    "Stop repeating — use the information you already have and proceed to "
                    "compile_test with your best draft."
                )
                trace_lines.append(f"  → {name}(repeated — nudged)")
                results.append({"role": "tool", "content": nudge})
                continue
            if verbose:
                display = {"code": "..."} if name == "compile_test" else args
                print(f"  tool: {name}({display})")
            result_text  = dispatch(name, args)
            result_lines = result_text.splitlines()
            if len(result_lines) > 150:
                result_text = (
                    "\n".join(result_lines[:150])
                    + f"\n... ({len(result_lines) - 150} more lines)"
                )
            trace_lines.append(
                f"  → {name}({json.dumps(args) if name != 'compile_test' else '{...}'})"
            )
            trace_lines.append(textwrap.indent(result_text, "      ") + "\n")
            results.append({"role": "tool", "content": result_text})
        return results

    def _extract_code(content: str) -> str:
        m = re.search(r"```(?:cpp|c\+\+)?\n(.*?)```", content, re.DOTALL)
        code = m.group(1).strip() if m else content.strip()
        _save_draft(code)
        return code

    def _is_usable(code: str | None) -> bool:
        return bool(code and len(code) > 100 and "#include" in code)

    for turn in range(20):
        data = _chat(messages, model, TEST_TOOLS)
        msg  = data.get("message", {})
        tool_calls        = msg.get("tool_calls") or []
        thinking          = msg.get("thinking",  "") or ""
        assistant_content = msg.get("content",   "") or ""

        messages.append({
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": tool_calls,
        })

        if thinking:
            if verbose:
                print(f"\n[thinking]\n{textwrap.indent(thinking, '  ')}\n")
            trace_lines.append(
                f"[thinking — turn {turn + 1}]\n{textwrap.indent(thinking, '  ')}\n"
            )

        if not tool_calls:
            trace_lines.append(
                f"[assistant — final]\n{textwrap.indent(assistant_content, '  ')}\n"
            )
            test_code = _extract_code(assistant_content)
            break

        trace_lines.append(
            f"[assistant — turn {turn + 1}, {len(tool_calls)} tool call(s)]"
        )
        messages.extend(_run_tool_calls(tool_calls))

    else:
        # All 20 turns used — ask once if more time is needed.
        trace_lines.append("[agent] Turn limit reached — asking for extension.\n")
        messages.append({"role": "user",
                         "content": _EXTENSION_PROMPT.format(n=_EXTENSION_TURNS)})

        data = _chat(messages, model, TEST_TOOLS)
        msg  = data.get("message", {})
        tool_calls        = msg.get("tool_calls") or []
        thinking          = msg.get("thinking",  "") or ""
        assistant_content = msg.get("content",   "") or ""
        messages.append({"role": "assistant", "content": assistant_content,
                         "tool_calls": tool_calls})

        if thinking:
            trace_lines.append(f"[thinking — extension]\n{textwrap.indent(thinking, '  ')}\n")

        if not tool_calls:
            trace_lines.append(
                f"[assistant — final (declined extension)]\n{textwrap.indent(assistant_content, '  ')}\n"
            )
            test_code = _extract_code(assistant_content)
        else:
            trace_lines.append(f"[agent] Extension granted ({_EXTENSION_TURNS} more turns).\n")
            messages.extend(_run_tool_calls(tool_calls))

            for ext_turn in range(_EXTENSION_TURNS):
                data = _chat(messages, model, TEST_TOOLS)
                msg  = data.get("message", {})
                tool_calls        = msg.get("tool_calls") or []
                thinking          = msg.get("thinking",  "") or ""
                assistant_content = msg.get("content",   "") or ""
                messages.append({"role": "assistant", "content": assistant_content,
                                 "tool_calls": tool_calls})

                if thinking:
                    trace_lines.append(
                        f"[thinking — ext turn {ext_turn + 1}]\n{textwrap.indent(thinking, '  ')}\n"
                    )

                if not tool_calls:
                    trace_lines.append(
                        f"[assistant — final]\n{textwrap.indent(assistant_content, '  ')}\n"
                    )
                    test_code = _extract_code(assistant_content)
                    break

                trace_lines.append(
                    f"[assistant — ext turn {ext_turn + 1}, {len(tool_calls)} tool call(s)]"
                )
                messages.extend(_run_tool_calls(tool_calls))
            else:
                trace_lines.append("[agent] Extension exhausted without final answer.\n")

    if not _is_usable(test_code):
        if _is_usable(last_draft[0]):
            trace_lines.append("[agent] Falling back to last saved draft.\n")
            test_code = last_draft[0]
        else:
            trace_lines.append("[agent] No usable output — issuing rescue prompt.\n")
            messages.append({"role": "user", "content": (
                "STOP. Do not call any tools. Output your best attempt at test.cc "
                "NOW inside a single ```cpp block. This is your last chance — any "
                "plausible draft is better than no output."
            )})
            try:
                data = _chat(messages, model, tools=[])
                assistant_content = (data.get("message") or {}).get("content") or ""
                trace_lines.append(
                    f"[assistant — rescue]\n{textwrap.indent(assistant_content, '  ')}\n"
                )
                rescued = _extract_code(assistant_content)
                if _is_usable(rescued):
                    test_code = rescued
                elif _is_usable(last_draft[0]):
                    test_code = last_draft[0]
            except (urllib.error.URLError, json.JSONDecodeError) as e:
                trace_lines.append(f"[agent] Rescue call failed: {e}\n")
                if _is_usable(last_draft[0]):
                    test_code = last_draft[0]

    return test_code, "\n".join(trace_lines)
