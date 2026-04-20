"""Generate a Google Test C++ file from an oracle directory."""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import urllib.request
from pathlib import Path

from ..runner.results import OracleResult, load_result, parse_output
from .compile import make_test_compile_cmd, write_compile_script

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL   = "gemma4:31b"

_SYSTEM_PROMPT = """\
You are converting a C++ oracle program into a Google Test suite.

Rules:
1. Keep all setup code (loading PDB/MTZ, constructing objects, calling the function) identical.
2. Replace every `std::cout << "OUTPUT ..." << std::endl;` with an EXPECT_EQ or EXPECT_TRUE/EXPECT_FALSE assertion.
   Use the observed values supplied by the user as the expected values.
3. Wrap everything in a single TEST(OracleTest, FunctionName) block.
4. Add the required Google Test headers and a main() that calls RUN_ALL_TESTS().
5. Remove all INPUT/OUTPUT std::cout lines — only keep the assertion logic.
6. Output only the complete C++ source in a single ```cpp block, no explanation.\
"""


def _ollama_chat(messages: list[dict], model: str) -> str:
    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(OLLAMA_CHAT_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def _extract_cpp(text: str) -> str:
    m = re.search(r"```cpp\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(#include.*)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _run_oracle(oracle_subdir: Path) -> OracleResult:
    binary = oracle_subdir / "oracle"
    proc = subprocess.run([str(binary)], capture_output=True, text=True)
    return parse_output(proc.returncode, proc.stdout, proc.stderr)



def _load_oracle_result(oracle_dir: Path) -> OracleResult:
    oracle_subdir = oracle_dir / "oracle"
    result_path = oracle_subdir / "result.json"
    if result_path.exists():
        return load_result(result_path)
    binary = oracle_subdir / "oracle"
    if not binary.exists():
        raise FileNotFoundError(
            f"oracle binary not found in {oracle_subdir} — compile first"
        )
    result = _run_oracle(oracle_subdir)
    if not result.success:
        raise RuntimeError(
            f"Oracle failed (exit {result.returncode}):\n{result.stderr[:500]}"
        )
    return result


def _write_test_files(oracle_dir: Path, test_src: str) -> Path:
    test_subdir = oracle_dir / "test"
    test_subdir.mkdir(exist_ok=True)
    test_cc = test_subdir / "test.cc"
    test_cc.write_text(test_src)
    write_compile_script(test_subdir)
    return test_cc


def generate_test(
    oracle_dir: Path,
    model: str = DEFAULT_MODEL,
    agent: bool = False,
    verbose: bool = False,
    conn: sqlite3.Connection | None = None,
) -> Path:
    """Generate test/test.cc from oracle/oracle.cc + observed outputs.

    Returns path to test.cc.
    """
    oracle_dir   = Path(oracle_dir)
    oracle_subdir = oracle_dir / "oracle"
    oracle_cc    = oracle_subdir / "oracle.cc"
    if not oracle_cc.exists():
        raise FileNotFoundError(f"oracle.cc not found in {oracle_subdir}")

    result = _load_oracle_result(oracle_dir)

    if agent:
        from ..db import connect
        from .agent import generate_test_with_agent

        test_subdir = oracle_dir / "test"
        _conn = conn or connect()
        try:
            test_src, trace = generate_test_with_agent(
                _conn, oracle_cc.read_text(), result.stdout,
                test_subdir=test_subdir, model=model, verbose=verbose,
            )
        finally:
            if conn is None:
                _conn.close()

        if test_src is None:
            raise RuntimeError("Agent produced no test code.")

        test_cc = _write_test_files(oracle_dir, test_src)
        (oracle_dir / "test" / "agent_trace.txt").write_text(trace)
        return test_cc

    # Non-agentic single-shot generation.
    user_msg = (
        f"Here is the oracle program:\n\n```cpp\n{oracle_cc.read_text()}\n```\n\n"
        f"Observed output when run:\n{result.stdout}\n\n"
        "Convert this into a Google Test suite using the observed values as expected values."
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    raw = _ollama_chat(messages, model)
    test_src = _extract_cpp(raw)
    return _write_test_files(oracle_dir, test_src)
