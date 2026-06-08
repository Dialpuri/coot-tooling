"""Print a human-readable summary of eval results.

Usage:
    python -m tooling.evaluate.summary                       # latest reports/eval-*.json
    python -m tooling.evaluate.summary reports/eval-2026-06-27.json
    python -m tooling.evaluate.summary generated-tests/      # scan every evaluate/ folder

When given a directory, the tool walks every function dir under it and reads the
`evaluate/<stage>.json` written for that function's current first-failing stage.
Unlike an aggregate `eval-*.json` (which only holds functions evaluated in a
single run), the scan includes every dir that has an evaluate record on disk —
so functions skipped on a re-run still count.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from .detect import first_failing_stage

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"

FAILURE_MODE_DESCRIPTIONS: dict[str, str] = {
    "compile_error_unfixed": "Compile errors the agent could not fix",
    "linker_error":          "Linker failures (missing symbols / libs)",
    "wrong_api_usage":       "Used non-existent API methods / types",
    "missing_type_info":     "Could not locate a required type or symbol",
    "bad_construction":      "Receiver or input arguments built incorrectly",
    "assertion_mismatch":    "Code ran but values disagreed with oracle",
    "degenerate_thinking":   "Cyclic reasoning / stream-abort halted progress",
    "ran_out_of_turns":      "Turn budget exhausted before a usable artefact",
    "gave_up":               "Agent explicitly surrendered",
    "never_compiled":        "Agent never invoked the compile tool",
    "infrastructure":        "External / environment failure (not model's fault)",
    "other":                 "Other / unclassified",
}

STAGES = ["oracle", "test", "gemmi"]


def _latest_report(reports_dir: Path) -> Path:
    candidates = sorted(reports_dir.glob("eval-*.json"))
    if not candidates:
        raise FileNotFoundError(f"No eval-*.json files found in {reports_dir}")
    return candidates[-1]


def _bar(n: int, total: int, width: int = 30) -> str:
    filled = round(width * n / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "0.0%"


def _print_tally(tally: dict[str, int], total: int, indent: str = "  ") -> None:
    sorted_items = sorted(tally.items(), key=lambda x: x[1], reverse=True)
    label_w = max((len(k) for k in tally), default=10)
    for mode, n in sorted_items:
        desc = FAILURE_MODE_DESCRIPTIONS.get(mode, mode)
        bar = _bar(n, total)
        print(f"{indent}{mode:<{label_w}}  {n:4d}  {_pct(n, total):6s}  {bar}  {desc}")


def _is_function_dir(d: Path) -> bool:
    """A generated-tests/<func>/ dir is one that has at least one stage trace."""
    return d.is_dir() and any((d / s / "agent_trace.txt").exists() for s in STAGES)


def _collect_from_dir(scan_dir: Path) -> tuple[list[dict], dict]:
    """Build a report dict by reading evaluate/<stage>.json from every function dir.

    For each function dir we take the json for its *current* first-failing stage,
    so the report reflects where each function stands now (and ignores stale jsons
    from a stage that has since been fixed). Returns (report_data, coverage).
    """
    results: list[dict] = []
    n_dirs = n_pass = n_fail = n_fail_no_eval = 0
    for d in sorted(scan_dir.iterdir()):
        if not _is_function_dir(d):
            continue
        n_dirs += 1
        fs = first_failing_stage(d)
        if fs is None:
            n_pass += 1
            continue
        n_fail += 1
        jf = d / "evaluate" / f"{fs.name}.json"
        if not jf.exists():
            n_fail_no_eval += 1
            continue
        try:
            results.append(json.loads(jf.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"warning: could not read {jf}: {e}", file=sys.stderr)
            n_fail_no_eval += 1
    coverage = {
        "scanned": n_dirs,
        "passing": n_pass,
        "failing": n_fail,
        "failing_with_eval": len(results),
        "failing_not_evaluated": n_fail_no_eval,
    }
    return results, coverage


def summarise(report_path: Path) -> None:
    data = json.loads(report_path.read_text())
    _summarise_data(data, title=f"Eval report: {report_path.name}")


def _build_report(results: list[dict], coverage: dict) -> dict:
    """Assemble a full report dict (same schema as reports/eval-*.json) so a scan
    can be saved and re-summarised later."""
    tally = dict(Counter(r["failure_mode"] for r in results))
    tally_by_stage: dict[str, dict[str, int]] = {}
    for r in results:
        tally_by_stage.setdefault(r["stage"], Counter())[r["failure_mode"]] += 1
    return {
        "results": results,
        "tally": tally,
        "tally_by_stage": {k: dict(v) for k, v in tally_by_stage.items()},
        "coverage": coverage,
    }


def summarise_dir(scan_dir: Path, *, out: Path | None = None) -> None:
    results, coverage = _collect_from_dir(scan_dir)
    report = _build_report(results, coverage)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"wrote {out}")
    _summarise_data(report, title=f"Eval scan: {scan_dir}", coverage=coverage)


def _summarise_data(data: dict, *, title: str, coverage: dict | None = None) -> None:
    results: list[dict] = data.get("results", [])
    tally: dict[str, int] = data.get("tally", {})
    tally_by_stage: dict[str, dict[str, int]] = data.get("tally_by_stage", {})

    total = len(results)
    if not tally:
        tally = dict(Counter(r["failure_mode"] for r in results))
    if not tally_by_stage:
        for r in results:
            tally_by_stage.setdefault(r["stage"], Counter())[r["failure_mode"]] += 1  # type: ignore[arg-type]

    stage_totals = {s: sum(tally_by_stage.get(s, {}).values()) for s in STAGES}
    total_evaluated = sum(stage_totals.values())

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print(title)
    print("=" * 70)
    if coverage is not None:
        print(f"  Function dirs scanned : {coverage['scanned']}")
        print(f"  Passing (all stages)  : {coverage['passing']}")
        print(f"  Failing               : {coverage['failing']}")
        print(f"    with eval record    : {coverage['failing_with_eval']}")
        print(f"    not yet evaluated   : {coverage['failing_not_evaluated']}")
    print(f"  Total failure records : {total}")
    print(f"  Breakdown by stage    : "
          + "  ".join(f"{s}={stage_totals[s]}" for s in STAGES if stage_totals[s]))
    print()

    # ── Overall tally ─────────────────────────────────────────────────────────
    print("── Overall failure-mode breakdown ──────────────────────────────────")
    _print_tally(tally, total_evaluated or total)
    print()

    # ── Per-stage tallies ─────────────────────────────────────────────────────
    for stage in STAGES:
        stage_tally = tally_by_stage.get(stage)
        if not stage_tally:
            continue
        n_stage = stage_totals[stage]
        print(f"── {stage.upper()} ({n_stage} failures) "
              + "─" * max(0, 50 - len(stage)))
        _print_tally(stage_tally, n_stage)
        print()

    # ── Confidence breakdown ──────────────────────────────────────────────────
    conf = Counter(r.get("confidence", "unknown") for r in results)
    if conf:
        print("── Confidence of classifications ───────────────────────────────────")
        for level in ("high", "medium", "low", "unknown"):
            n = conf.get(level, 0)
            if n:
                print(f"  {level:<8s} {n:4d}  {_pct(n, total):6s}")
        print()

    # ── Top recurring notes (first sentence) per failure mode ────────────────
    print("── Most common root causes (top 3 notes per failure mode) ──────────")
    from collections import defaultdict
    notes_by_mode: dict[str, list[str]] = defaultdict(list)
    for r in results:
        note = (r.get("note") or "").strip()
        if note:
            first_sentence = note.split(".")[0].strip()
            notes_by_mode[r["failure_mode"]].append(first_sentence)

    sorted_modes = sorted(tally.items(), key=lambda x: x[1], reverse=True)
    for mode, _ in sorted_modes[:8]:
        sentences = notes_by_mode.get(mode, [])
        if not sentences:
            continue
        freq = Counter(sentences).most_common(3)
        print(f"\n  {mode}:")
        for sentence, cnt in freq:
            trimmed = sentence[:110] + "…" if len(sentence) > 110 else sentence
            print(f"    ({cnt:3d}×) {trimmed}")
    print()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="tooling.evaluate.summary")
    ap.add_argument("path", nargs="?", type=Path,
                    help="A reports/eval-*.json file, or a directory "
                         "(e.g. generated-tests/) to scan for evaluate/ folders. "
                         "Defaults to the latest reports/eval-*.json.")
    ap.add_argument("-o", "--out", type=Path, default=None, metavar="FILE",
                    help="When scanning a directory, also write the assembled "
                         "report JSON here (re-summarisable like reports/eval-*.json).")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    path = args.path
    if path is None:
        try:
            path = _latest_report(REPORTS_DIR)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    elif not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2

    if path.is_dir():
        summarise_dir(path, out=args.out)
    else:
        if args.out is not None:
            print("note: --out is ignored when summarising an existing report file",
                  file=sys.stderr)
        summarise(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
