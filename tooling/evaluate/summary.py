"""Print a human-readable summary of the latest eval report in reports/.

Usage:
    python -m tooling.evaluate.summary
    python -m tooling.evaluate.summary reports/eval-2026-06-27.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

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


def summarise(report_path: Path) -> None:
    data = json.loads(report_path.read_text())
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
    print(f"Eval report: {report_path.name}")
    print("=" * 70)
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
    args = sys.argv[1:] if argv is None else argv
    if args:
        path = Path(args[0])
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 2
    else:
        try:
            path = _latest_report(REPORTS_DIR)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    summarise(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
