"""Orchestrate a single per-port lift: run the agent, verify, persist.

Artifacts land in `<port_dir>/reconstitute/`:
    function.hh / function.cc / test.cc   the lifted method + adapted test
    agent_trace.txt                        full lift agent trace
    reconstitute_ok                        marker written only on a verified pass
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from ..db import connect
from .agent import run_lift_agent, DEFAULT_MODEL
from .agent_reexports import dep_extra_includes, dep_extra_sources
from .classmodel import class_model_for
from ..gemmi.compile import compile_gemmi, run_gemmi_test_binary


class SourcePortNotPassing(Exception):
    """The phase-1 gemmi port we were asked to lift no longer compiles/passes.

    Raised before the agent runs so callers can SKIP (not fail) the port — a
    broken input is not the lift's fault, and asking the model to "mechanically
    move" code that doesn't compile is a contradiction it cannot satisfy.
    Usually caused by dependency drift in a `_gemmi` callee the port relies on.
    """


def _port_is_passing(gemmi_dir: Path) -> bool:
    return (gemmi_dir / "function.hh").exists() and (gemmi_dir / "test.cc").exists()


def verify_source_port(
    gemmi_dir: Path,
    dep_includes: list[Path],
    dep_sources: list[Path],
) -> tuple[bool, str]:
    """Compile + run the phase-1 gemmi port as-is. Returns (passed, log)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        for f in ("function.hh", "function.cc", "test.cc"):
            if (gemmi_dir / f).exists():
                (tp / f).write_text((gemmi_dir / f).read_text())
        fcc = tp / "function.cc"
        ok, out = compile_gemmi(tp / "test.cc", tp / "test",
                                fcc if fcc.exists() else None,
                                dep_includes, dep_sources)
        if not ok:
            return False, out
        ok, out = run_gemmi_test_binary(tp / "test")
        return ok, out


def lift_one(
    port_dir: Path,
    class_qname: str,
    method_qname: str,
    sig_hash: str | None = None,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    conn: sqlite3.Connection | None = None,
    verify_source: bool = True,
) -> Path | None:
    """Lift the verified gemmi port in `port_dir` into a `<class>_gemmi` method.

    Returns the path to the persisted reconstitute/test.cc on success, else None.
    Raises SourcePortNotPassing if `verify_source` and the phase-1 port no longer
    compiles/passes — a broken input the lift can't be expected to fix.
    """
    port_dir = Path(port_dir)
    gemmi_dir = port_dir / "gemmi"
    if not _port_is_passing(gemmi_dir):
        raise FileNotFoundError(
            f"No verified gemmi port at {gemmi_dir} (need function.hh + test.cc)"
        )

    cm = class_model_for(class_qname)
    work_dir = port_dir / "reconstitute"

    _conn = conn or connect()
    try:
        dep_includes = dep_extra_includes(_conn, method_qname)
        dep_sources = dep_extra_sources(_conn, method_qname)

        # Gate: never ask the agent to "mechanically move" code that doesn't
        # compile. Cheap source compile up front avoids burning ~10 LLM turns
        # on a stale port (e.g. one whose _gemmi deps have drifted).
        if verify_source:
            ok, log = verify_source_port(gemmi_dir, dep_includes, dep_sources)
            if not ok:
                work_dir.mkdir(parents=True, exist_ok=True)
                (work_dir / "source_broken.log").write_text(log)
                raise SourcePortNotPassing(
                    f"{method_qname}: phase-1 gemmi port no longer compiles/passes"
                )

        blocks, trace = run_lift_agent(
            _conn,
            class_model=cm,
            method_qname=method_qname,
            gemmi_dir=gemmi_dir,
            work_dir=work_dir,
            dep_includes=dep_includes,
            dep_sources=dep_sources,
            model=model,
            verbose=verbose,
        )
    finally:
        if conn is None:
            _conn.close()

    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "agent_trace.txt").write_text(trace)

    if not blocks or "function.hh" not in blocks or "test.cc" not in blocks:
        print(f"[lift] {method_qname}: agent produced no usable output")
        return None

    # Verify the lift independently before persisting — same gate as phase 1.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "function.hh").write_text(blocks["function.hh"])
        (tmp_path / "test.cc").write_text(blocks["test.cc"])
        has_cc = bool(blocks.get("function.cc", "").strip())
        if has_cc:
            (tmp_path / "function.cc").write_text(blocks["function.cc"])

        ok, output = compile_gemmi(
            tmp_path / "test.cc", tmp_path / "test",
            (tmp_path / "function.cc") if has_cc else None,
            dep_includes, dep_sources,
        )
        (work_dir / "compile.log").write_text(output)
        if not ok:
            print(f"[lift] {method_qname}: verify compile FAILED")
            return None
        ok, output = run_gemmi_test_binary(tmp_path / "test")
        (work_dir / "run.log").write_text(output)
        if not ok:
            print(f"[lift] {method_qname}: verify test FAILED")
            return None

    # Passed — persist final files.
    (work_dir / "function.hh").write_text(blocks["function.hh"])
    test_cc = work_dir / "test.cc"
    test_cc.write_text(blocks["test.cc"])
    if blocks.get("function.cc", "").strip():
        (work_dir / "function.cc").write_text(blocks["function.cc"])
    elif (work_dir / "function.cc").exists():
        (work_dir / "function.cc").unlink()
    (work_dir / "reconstitute_ok").write_text("")
    print(f"[lift] {method_qname}: PASS")
    return test_cc
