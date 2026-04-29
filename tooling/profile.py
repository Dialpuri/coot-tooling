"""Profile test run successes and failures across generated-tests.

Scans every function directory under generated-tests/ and parses the
GoogleTest run.log in each stage subdirectory (oracle, test, gemmi).

Usage:
  python -m tooling.profile
  python -m tooling.profile --stage test
  python -m tooling.profile --failures
  python -m tooling.profile --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

OUT_ROOT = Path(__file__).parent.parent / "generated-tests"

STAGES = ("oracle", "test", "gemmi")

_PASSED_RE = re.compile(r"\[\s*PASSED\s*\]\s+(\d+) test")
_FAILED_RE = re.compile(r"\[\s*FAILED\s*\]\s+(\d+) test")
_FAIL_NAME_RE = re.compile(r"\[\s*FAILED\s*\]\s+\S+\.\S+")


def _log_passed(log_path: Path) -> bool | None:
    """Return True=pass, False=fail, None=missing/incomplete from a GoogleTest run.log."""
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    m_f = _FAILED_RE.search(text)
    if m_f and int(m_f.group(1)) > 0:
        return False
    m_p = _PASSED_RE.search(text)
    if m_p and int(m_p.group(1)) > 0:
        return True
    return None


def _status_oracle(stage_dir: Path) -> str:
    """pass = result.json exists with at least one case; fail = exists but no cases; missing = absent."""
    result_json = stage_dir / "result.json"
    if not result_json.exists():
        return "missing"
    try:
        data = json.loads(result_json.read_text())
        return "pass" if data.get("cases") else "fail"
    except Exception:
        return "fail"


def _status_test(stage_dir: Path) -> str:
    """pass/fail from run.log PASSED/FAILED line; missing if log absent or incomplete."""
    result = _log_passed(stage_dir / "run.log")
    if result is True:
        return "pass"
    if result is False:
        return "fail"
    return "missing"


def _status_gemmi(stage_dir: Path) -> str:
    """pass = function.hh + test.cc exist AND run.log passes; fail = files exist but log fails; missing otherwise."""
    has_files = (stage_dir / "function.hh").exists() and (stage_dir / "test.cc").exists()
    if not has_files:
        return "missing"
    result = _log_passed(stage_dir / "run.log")
    if result is True:
        return "pass"
    if result is False:
        return "fail"
    return "missing"


_STATUS_FN = {
    "oracle": _status_oracle,
    "test": _status_test,
    "gemmi": _status_gemmi,
}


@dataclass
class FunctionResult:
    name: str
    stage_status: dict[str, str] = field(default_factory=dict)   # stage -> pass/fail/missing
    failure_details: dict[str, list[str]] = field(default_factory=dict)  # stage -> lines


def _collect_failure_lines(log_path: Path) -> list[str]:
    lines = []
    for line in log_path.read_text(errors="replace").splitlines():
        if "Failure" in line or _FAIL_NAME_RE.match(line.strip()):
            lines.append(line.rstrip())
    return lines


def collect(stages: list[str]) -> list[FunctionResult]:
    results = []
    for fn_dir in sorted(OUT_ROOT.iterdir()):
        if not fn_dir.is_dir() or fn_dir.name.startswith("_"):
            continue
        r = FunctionResult(name=fn_dir.name)
        for stage in stages:
            stage_dir = fn_dir / stage
            status = _STATUS_FN[stage](stage_dir)
            r.stage_status[stage] = status
            if status == "fail":
                log = stage_dir / "run.log"
                if log.exists():
                    r.failure_details[stage] = _collect_failure_lines(log)
        results.append(r)
    return results


def _hr(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def _stage_counts(results: list[FunctionResult], stage: str) -> tuple[int, int, int]:
    n_pass = sum(1 for r in results if r.stage_status.get(stage) == "pass")
    n_fail = sum(1 for r in results if r.stage_status.get(stage) == "fail")
    n_miss = sum(1 for r in results if r.stage_status.get(stage) == "missing")
    return n_pass, n_fail, n_miss


def print_report(results: list[FunctionResult], stages: list[str], failures_only: bool) -> None:
    total = len(results)

    if not failures_only:
        _hr(f"Per-function status ({total} functions)")
        col_w = max(len(r.name) for r in results)
        stage_w = 8
        hdr = f"  {'function':<{col_w}}" + "".join(f"  {s:^{stage_w}}" for s in stages)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in results:
            row = f"  {r.name:<{col_w}}"
            for stage in stages:
                st = r.stage_status.get(stage, "missing")
                sym = {"pass": "OK", "fail": "FAIL", "missing": "—"}.get(st, st)
                row += f"  {sym:^{stage_w}}"
            print(row)

    _hr("Failing tests")
    any_fail = False
    for r in results:
        for stage in stages:
            if r.stage_status.get(stage) == "fail":
                any_fail = True
                print(f"\n  [{stage}] {r.name}")
                for line in r.failure_details.get(stage, [])[:10]:
                    print(f"    {line}")
    if not any_fail:
        print("  (none)")

    _hr("Stage summary")
    header = f"  {'stage':<10}  {'pass':>6}  {'fail':>6}  {'missing':>8}  {'pass%':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for stage in stages:
        n_pass, n_fail, n_miss = _stage_counts(results, stage)
        ran = n_pass + n_fail
        pct = f"{100*n_pass/ran:.1f}%" if ran else "—"
        print(f"  {stage:<10}  {n_pass:>6}  {n_fail:>6}  {n_miss:>8}  {pct:>6}")



def write_csv(results: list[FunctionResult], stages: list[str], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["function"] + stages)
        for r in results:
            w.writerow([r.name] + [r.stage_status.get(s, "missing") for s in stages])
    print(f"CSV written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=STAGES, default=None,
                        help="restrict to a single stage")
    parser.add_argument("--failures", action="store_true",
                        help="only show failure details, skip per-function table")
    parser.add_argument("--csv", metavar="FILE", default=None,
                        help="also write results to a CSV file")
    args = parser.parse_args()

    stages = [args.stage] if args.stage else list(STAGES)
    results = collect(stages)

    print(f"Generated-tests root: {OUT_ROOT}")
    print(f"Functions found: {len(results)}  |  Stages: {', '.join(stages)}")

    print_report(results, stages, failures_only=args.failures)

    if args.csv:
        write_csv(results, stages, args.csv)


if __name__ == "__main__":
    main()
