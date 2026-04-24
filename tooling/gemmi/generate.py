"""Drive the combined gemmi port + test generation."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ..db import connect, get_function
from .compile import write_compile_script

DEFAULT_MODEL = "qwen3:30b"


def _write_files(oracle_dir: Path, blocks: dict[str, str]) -> Path:
    gemmi_subdir = oracle_dir / "gemmi"
    gemmi_subdir.mkdir(exist_ok=True)
    (gemmi_subdir / "function.hh").write_text(blocks["function.hh"])
    (gemmi_subdir / "test.cc").write_text(blocks["test.cc"])
    has_cc = "function.cc" in blocks and blocks["function.cc"].strip()
    if has_cc:
        (gemmi_subdir / "function.cc").write_text(blocks["function.cc"])
    write_compile_script(gemmi_subdir, has_function_cc=bool(has_cc))
    return gemmi_subdir / "test.cc"


def generate_gemmi(
    oracle_dir: Path,
    function_qname: str,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    conn: sqlite3.Connection | None = None,
) -> Path:
    """Emit oracle_dir/gemmi/{function.hh, [function.cc], test.cc}.

    Requires oracle_dir/test/test.cc to exist (the MMDB test whose
    assertions are carried over unchanged).
    """
    from .agent import generate_gemmi_port_with_agent

    oracle_dir = Path(oracle_dir)
    original_test = oracle_dir / "test" / "test.cc"
    if not original_test.exists():
        raise FileNotFoundError(
            f"MMDB test not found at {original_test} — run tooling.test first"
        )

    gemmi_subdir = oracle_dir / "gemmi"
    _conn = conn or connect()
    try:
        row = get_function(_conn, function_qname)
        if row is None or not row["source_code"]:
            raise RuntimeError(
                f"No source found in code_graph.db for {function_qname}"
            )
        blocks, trace = generate_gemmi_port_with_agent(
            _conn,
            original_function_src=row["source_code"],
            function_qname=function_qname,
            original_test_cc=original_test.read_text(),
            gemmi_subdir=gemmi_subdir,
            model=model,
            verbose=verbose,
        )
    finally:
        if conn is None:
            _conn.close()

    if blocks is None:
        raise RuntimeError("Agent produced no usable port.")

    test_cc = _write_files(oracle_dir, blocks)
    (gemmi_subdir / "agent_trace.txt").write_text(trace)
    return test_cc
