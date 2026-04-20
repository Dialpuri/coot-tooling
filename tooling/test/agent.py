"""Agentic Google Test generation — model calls tools to resolve headers,
look up types, and iteratively compile its draft before finalising."""
from __future__ import annotations

import json
import re
import sqlite3
import textwrap
import urllib.request
from pathlib import Path

from ..oracle.agent import OLLAMA_CHAT_URL, TOOLS, _dispatch
from .compile import MAX_COMPILE_ATTEMPTS, compile_test_cc, run_test_binary

TEST_SYSTEM_PROMPT = """\
You are converting a C++ oracle program into a Google Test suite (test.cc).

Rules:
1. Keep all setup code (loading PDB/MTZ, constructing objects, calling the
   function) identical to the oracle.
2. Replace every `std::cout << "OUTPUT ..." << std::endl;` with an
   EXPECT_EQ or EXPECT_TRUE/EXPECT_FALSE assertion using the observed values.
3. Wrap the body in a single TEST(OracleTest, <FunctionName>) block.
4. Add a main() that calls RUN_ALL_TESTS().
5. Remove all INPUT/OUTPUT std::cout lines — assertions only.
6. Call resolve_includes on your draft to verify every #include "..." header
   resolves correctly. Fix any that do not.
7. Call compile_test with your draft. Fix all compiler errors and call again
   until it succeeds (max 3 attempts).
8. Once it compiles, call run_test to check for assertion failures. Fix any
   EXPECT_EQ mismatches (wrong expected values, type issues) and recompile.
9. Output only the final, compiling and passing C++ source in a single ```cpp block.\
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

TEST_TOOLS = TOOLS + [_COMPILE_TEST_TOOL, _RUN_TEST_TOOL]


def _make_tool_handlers(test_subdir: Path) -> tuple[callable, callable]:
    """Return (compile_handler, run_handler) sharing state about the last build."""
    attempts    = [0]
    last_binary = [None]   # Path | None — set when a compile succeeds

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

        lines = output.splitlines()
        if len(lines) > 100:
            output = "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines)"

        if success:
            last_binary[0] = test_bin
            return f"Compilation succeeded (attempt {attempts[0]}/{MAX_COMPILE_ATTEMPTS})."
        last_binary[0] = None
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

    return compile_handler, run_handler


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
    oracle_stdout: str,
    test_subdir: Path,
    model: str,
    verbose: bool = False,
) -> tuple[str | None, str]:
    """Run the agentic test generation loop.

    Returns (test_code, trace_text).
    test_code is None if the model produced nothing usable.
    """
    compile_handler, run_handler = _make_tool_handlers(test_subdir)

    def dispatch(name: str, args: dict) -> str:
        if name == "compile_test":
            return compile_handler(args["code"])
        if name == "run_test":
            return run_handler()
        return _dispatch(conn, name, args)

    user_content = (
        f"Here is the oracle program:\n\n```cpp\n{oracle_cc_text}\n```\n\n"
        f"Observed output when run:\n{oracle_stdout}\n"
        "There may be errors and warnings at the top about file paths, ignore them unless "
        "there is no other output.\n\n"
        "Convert this into a Google Test suite using the observed values as "
        "expected values. Use the tools to verify headers, look up any types "
        "you are unsure about, and compile your draft before finalising."
    )

    messages: list[dict] = [
        {"role": "system", "content": TEST_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    trace_lines: list[str] = [
        "=== TEST AGENT TRACE ===\n",
        f"[user]\n{textwrap.indent(user_content, '  ')}\n",
    ]

    max_turns = 20
    test_code: str | None = None

    for turn in range(max_turns):
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
            m = re.search(r"```(?:cpp|c\+\+)?\n(.*?)```", assistant_content, re.DOTALL)
            test_code = m.group(1).strip() if m else assistant_content.strip()
            break

        trace_lines.append(
            f"[assistant — turn {turn + 1}, {len(tool_calls)} tool call(s)]"
        )
        tool_results: list[dict] = []

        for call in tool_calls:
            fn_info = call.get("function", {})
            name    = fn_info.get("name", "")
            args    = fn_info.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            if verbose:
                display = {'code': '...'} if name == 'compile_test' else args
                print(f"  tool: {name}({display})")

            result_text = dispatch(name, args)

            result_lines = result_text.splitlines()
            if len(result_lines) > 150:
                result_text = (
                    "\n".join(result_lines[:150])
                    + f"\n... ({len(result_lines) - 150} more lines)"
                )

            trace_lines.append(f"  → {name}({json.dumps(args) if name != 'compile_test' else '{...}'})")
            trace_lines.append(textwrap.indent(result_text, "      ") + "\n")

            tool_results.append({"role": "tool", "content": result_text})

        messages.extend(tool_results)

    else:
        trace_lines.append("[agent] Max turns reached without a final answer.\n")

    return test_code, "\n".join(trace_lines)
