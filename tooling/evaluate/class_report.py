"""Show evaluation results for all functions belonging to a specific class.

Usage:
    python -m tooling.evaluate.class_report coot::molecule_t
    python -m tooling.evaluate.class_report coot::restraints_container_t --stage gemmi
    python -m tooling.evaluate.class_report coot::util --group-by failure_mode
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .class_progress import _build_class_map, _is_function_dir, _qname_from_dir
from .detect import stage_statuses, STAGES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESTS_DIR = REPO_ROOT / "generated-tests"

CONFIDENCE_SYMBOL = {"high": "●", "medium": "◑", "low": "○", "unknown": "?"}


def _load_eval(func_dir: Path, stage: str) -> dict | None:
    p = func_dir / "evaluate" / f"{stage}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _short_name(qname: str, class_qname: str) -> str:
    """Strip the class prefix so output is easier to scan."""
    prefix = class_qname + "::"
    if qname.startswith(prefix):
        return qname[len(prefix):]
    return qname


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tooling.evaluate.class_report",
                                 description="Evaluation details for all functions in a class.")
    ap.add_argument("class_name", metavar="CLASS",
                    help="Fully-qualified class name, e.g. coot::molecule_t.")
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT_TESTS_DIR,
                    help=f"Root dir containing function subdirs (default: {DEFAULT_TESTS_DIR}).")
    ap.add_argument("--stage", choices=STAGES, default=None,
                    help="Restrict output to one pipeline stage.")
    ap.add_argument("--group-by", choices=("function", "failure_mode"), default="function",
                    dest="group_by",
                    help="Organise rows by function (default) or grouped by failure mode.")
    ap.add_argument("--no-passed", action="store_true",
                    help="Hide functions where every attempted stage passed.")
    args = ap.parse_args(argv)

    if not args.path.exists():
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 2

    func_dirs = sorted(d for d in args.path.iterdir() if _is_function_dir(d))
    if not func_dirs:
        print(f"error: no function dirs found under {args.path}", file=sys.stderr)
        return 2

    class_map = _build_class_map(func_dirs)
    target = args.class_name
    matched = [d for d in func_dirs if class_map[d] == target]

    if not matched:
        print(f"error: no functions found for class '{target}'", file=sys.stderr)
        print("Hint: run class_progress to list available classes.", file=sys.stderr)
        return 2

    stages_to_show = [args.stage] if args.stage else list(STAGES)

    # Collect per-function records
    # record: {qname, stage, status, eval: dict|None}
    records: list[dict] = []
    for func_dir in matched:
        qname = _qname_from_dir(func_dir)
        statuses = {s.name: s for s in stage_statuses(func_dir)}
        for stage in stages_to_show:
            s = statuses.get(stage)
            if s is None or not s.present:
                continue
            ev = _load_eval(func_dir, stage)
            records.append({
                "qname": qname,
                "short": _short_name(qname, target),
                "stage": stage,
                "passed": s.passed,
                "reason": s.reason,
                "eval": ev,
            })

    if args.no_passed:
        records = [r for r in records if not r["passed"]]

    # ── Header ────────────────────────────────────────────────────────────────
    total_fns = len(matched)
    failed_fns = sum(
        1 for d in matched
        if any(
            not s.passed
            for s in stage_statuses(d)
            if s.present and s.name in stages_to_show
        )
    )
    print()
    print(f"Class: {target}")
    print(f"Functions: {total_fns} total, {failed_fns} with at least one stage failure")
    if stages_to_show != list(STAGES):
        print(f"Stage filter: {', '.join(stages_to_show)}")
    print("=" * 72)

    if args.group_by == "failure_mode":
        _print_by_failure_mode(records, target)
    else:
        _print_by_function(records, target, stages_to_show)

    # ── Tally ─────────────────────────────────────────────────────────────────
    failed_records = [r for r in records if not r["passed"]]
    if failed_records:
        print()
        print("── Failure-mode tally " + "─" * 50)
        tally: Counter[str] = Counter()
        for r in failed_records:
            mode = (r["eval"] or {}).get("failure_mode", "not evaluated")
            tally[mode] += 1
        total = sum(tally.values())
        name_w = max(len(k) for k in tally)
        for mode, n in tally.most_common():
            bar_filled = round(20 * n / total) if total else 0
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            pct = f"{100 * n / total:.1f}%"
            print(f"  {mode:<{name_w}}  {n:4d}  {pct:6s}  {bar}")

    print()
    return 0


def _print_by_function(records: list[dict], target: str,
                        stages_to_show: list[str]) -> None:
    # Group by qname so we print all stages for each function together
    by_fn: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_fn[r["qname"]].append(r)

    for qname in sorted(by_fn):
        fn_records = by_fn[qname]
        short = _short_name(qname, target)
        all_passed = all(r["passed"] for r in fn_records)
        status_icon = "✓" if all_passed else "✗"
        print(f"\n{status_icon} {short}")

        for r in sorted(fn_records, key=lambda x: STAGES.index(x["stage"])):
            _print_record_line(r, indent="  ")


def _print_by_failure_mode(records: list[dict], target: str) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["passed"]:
            continue
        mode = (r["eval"] or {}).get("failure_mode", "not evaluated")
        by_mode[mode].append(r)

    for mode in sorted(by_mode, key=lambda m: -len(by_mode[m])):
        group = by_mode[mode]
        print(f"\n── {mode} ({len(group)}) " + "─" * max(0, 50 - len(mode)))
        for r in sorted(group, key=lambda x: (x["short"], x["stage"])):
            short = r["short"]
            stage = r["stage"]
            ev = r["eval"] or {}
            conf = CONFIDENCE_SYMBOL.get(ev.get("confidence", ""), "?")
            note = ev.get("note", r["reason"])
            print(f"  {conf} [{stage:6s}] {short}")
            if note:
                # Wrap note at 90 chars
                words = note.split()
                line, lines = [], []
                for w in words:
                    if sum(len(x) + 1 for x in line) + len(w) > 88:
                        lines.append(" ".join(line))
                        line = [w]
                    else:
                        line.append(w)
                if line:
                    lines.append(" ".join(line))
                for l in lines:
                    print(f"           {l}")


def _print_record_line(r: dict, indent: str = "") -> None:
    stage = r["stage"]
    ev = r["eval"] or {}
    if r["passed"]:
        print(f"{indent}[{stage:6s}] ✓ passed")
        return
    mode = ev.get("failure_mode", "not evaluated")
    conf = CONFIDENCE_SYMBOL.get(ev.get("confidence", ""), "?")
    note = ev.get("note", r["reason"])
    print(f"{indent}[{stage:6s}] {conf} {mode}")
    if note:
        words = note.split()
        line, lines = [], []
        for w in words:
            if sum(len(x) + 1 for x in line) + len(w) > 80 - len(indent):
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        for l in lines:
            print(f"{indent}         {l}")


if __name__ == "__main__":
    raise SystemExit(main())
