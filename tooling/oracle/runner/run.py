"""Run compiled oracle binaries and return structured results."""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from .results import (
    OracleResult, OraclePanelResult, parse_output, save_result, save_panel_result,
)
from ..compile import ccp4_env
from ..pdb_selector import fixture_panel


RUN_TIMEOUT_SECONDS = 20


def run_binary(binary: Path, args: list[str] | None = None, cwd: Path | None = None,
               attempts: int = 2) -> tuple[int, str, str]:
    """Run a binary, return (returncode, stdout, stderr).

    Retries once on timeout — the hang appears non-deterministic and a fresh
    invocation typically completes in milliseconds.
    """
    cmd = [str(binary.absolute())] + (args or [])
    cwd_str = str(cwd) if cwd else str(binary.parent)
    last_err = ""
    for attempt in range(1, attempts + 1):
        # Capture BYTES, not text: oracle/coot output carries non-UTF-8 bytes
        # (e.g. 0xb0 '°' in cell angles), which `text=True` would raise
        # UnicodeDecodeError on. Decode with errors="replace" instead.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=cwd_str, start_new_session=True, env=ccp4_env(),
        )
        try:
            out_b, err_b = proc.communicate(timeout=RUN_TIMEOUT_SECONDS)
            return (
                proc.returncode,
                out_b.decode("utf-8", errors="replace"),
                err_b.decode("utf-8", errors="replace"),
            )
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            last_err = f"[run_binary] timed out after {RUN_TIMEOUT_SECONDS}s (attempt {attempt}/{attempts})"
    return -1, "", last_err


def run_oracle(oracle_dir: Path) -> OracleResult:
    """Run the compiled oracle in oracle_dir, parse its output, and save result.json."""
    binary = oracle_dir / "oracle"
    if not binary.exists():
        return OracleResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr=f"Binary not found: {binary}",
            inputs={},
            outputs={},
        )

    returncode, stdout, stderr = run_binary(binary, cwd=oracle_dir)
    result = parse_output(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )

    # Save stdout/stderr logs alongside the binary.
    (oracle_dir / "run.log").write_text(stdout + stderr)
    save_result(oracle_dir / "result.json", result)

    return result


def _fixture_basename(structure_path: str) -> str:
    return Path(structure_path).name


def run_oracle_panel(
    oracle_dir: Path,
    fixtures: list[tuple[str, str | None]] | None = None,
) -> OraclePanelResult:
    """Run the compiled oracle across the multi-fixture panel.

    The oracle binary takes argv[1]=structure path and argv[2]=mtz path (empty
    string when a fixture has no paired map). Each fixture is parsed into its own
    OracleResult; per-fixture logs are written as `run-<fixture>.log` and the
    combined panel is saved to `panel.json`. For back-compat the primary passing
    fixture is also written to `run.log` / `result.json`.
    """
    binary = oracle_dir / "oracle"
    fixtures = fixtures if fixtures is not None else fixture_panel()
    panel = OraclePanelResult()

    if not binary.exists():
        for structure_path, _ in fixtures:
            panel.per_fixture[_fixture_basename(structure_path)] = OracleResult(
                success=False, returncode=-1, stdout="",
                stderr=f"Binary not found: {binary}", inputs={}, outputs={},
            )
        save_panel_result(oracle_dir / "panel.json", panel)
        return panel

    for structure_path, mtz_path in fixtures:
        name = _fixture_basename(structure_path)
        rc, stdout, stderr = run_binary(
            binary, args=[structure_path, mtz_path or ""], cwd=oracle_dir,
        )
        result = parse_output(returncode=rc, stdout=stdout, stderr=stderr)
        panel.per_fixture[name] = result
        (oracle_dir / f"run-{name}.log").write_text(stdout + stderr)

    save_panel_result(oracle_dir / "panel.json", panel)
    # Mirror the primary passing fixture into the legacy single-result files so
    # downstream coverage/notes that still read result.json keep working.
    primary = panel.primary()
    if primary is not None:
        (oracle_dir / "run.log").write_text(primary.stdout + primary.stderr)
        save_result(oracle_dir / "result.json", primary)

    return panel
