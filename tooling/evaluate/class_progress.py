"""Show per-class pass/fail percentages across pipeline stages.

Usage:
    python -m tooling.evaluate.class_progress
    python -m tooling.evaluate.class_progress generated-tests/
    python -m tooling.evaluate.class_progress generated-tests/ --stage gemmi
    python -m tooling.evaluate.class_progress generated-tests/ --min-functions 3
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from .detect import stage_statuses, STAGES
from ..db import connect

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESTS_DIR = REPO_ROOT / "generated-tests"

_CLASS_KINDS = frozenset({
    "CLASS_DECL", "CLASS_TEMPLATE", "STRUCT_DECL",
    "CLASS_TEMPLATE_PARTIAL_SPECIALIZATION",
})


def _qname_from_dir(d: Path) -> str:
    return d.name.replace("__", "::")


def _outermost_class(conn, qname: str) -> str:
    """Return the outermost enclosing class for a function qualified name.

    Walks the :: chain from the outside in; returns the first prefix that
    exists in the types table as a class or struct.  Namespaces are not in
    the types table, so they are skipped automatically.  Falls back to
    stripping the last component when no class ancestor is found.
    """
    parts = qname.split("::")
    for i in range(1, len(parts)):
        candidate = "::".join(parts[:i])
        row = conn.execute(
            "SELECT kind FROM types WHERE qualified_name = ? LIMIT 1",
            (candidate,),
        ).fetchone()
        if row and row["kind"] in _CLASS_KINDS:
            return candidate
    if "::" in qname:
        return qname.rsplit("::", 1)[0]
    return "(top-level)"


def _build_class_map(func_dirs: list[Path]) -> dict[Path, str]:
    """Return {func_dir: class_name} using a single DB connection."""
    conn = connect()
    try:
        return {d: _outermost_class(conn, _qname_from_dir(d)) for d in func_dirs}
    finally:
        conn.close()


def _is_function_dir(d: Path) -> bool:
    return d.is_dir() and any((d / s / "agent_trace.txt").exists() for s in STAGES)


def _bar(n: int, total: int, width: int = 20) -> str:
    filled = round(width * n / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:5.1f}%" if total else "  n/a "


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tooling.evaluate.class_progress",
                                 description="Per-class pass percentages across pipeline stages.")
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT_TESTS_DIR,
                    help=f"Root dir containing function subdirs (default: {DEFAULT_TESTS_DIR}).")
    ap.add_argument("--stage", choices=STAGES, default=None,
                    help="Show breakdown for one stage only.")
    ap.add_argument("--min-functions", type=int, default=1, metavar="N",
                    help="Only show classes with at least N functions (default: 1).")
    ap.add_argument("--sort", choices=("class", "pct", "total", "functions"), default="class",
                    help="Sort rows by class name, pass percentage, total stage attempts, "
                         "or number of functions in the class (default: class).")
    args = ap.parse_args(argv)

    if not args.path.exists():
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 2

    func_dirs = sorted(d for d in args.path.iterdir() if _is_function_dir(d))
    if not func_dirs:
        print(f"error: no function dirs found under {args.path}", file=sys.stderr)
        return 2

    stages_to_show = [args.stage] if args.stage else list(STAGES)

    class_map = _build_class_map(func_dirs)

    # class → stage → (passed, attempted)
    counts: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {s: [0, 0] for s in STAGES}
    )

    for func_dir in func_dirs:
        cls = class_map[func_dir]
        for status in stage_statuses(func_dir):
            if status.present:
                counts[cls][status.name][1] += 1  # attempted
                if status.passed:
                    counts[cls][status.name][0] += 1  # passed

    # Filter by min-functions (use oracle as proxy for "attempted at all")
    filtered = {
        cls: data for cls, data in counts.items()
        if sum(data[s][1] for s in STAGES) >= args.min_functions
    }

    if not filtered:
        print("No classes meet the --min-functions threshold.", file=sys.stderr)
        return 1

    # Sorting
    def _sort_key(item: tuple[str, dict]) -> tuple:
        cls, data = item
        if args.sort == "pct":
            # Use the last shown stage's pass % as sort key (descending → negate)
            s = stages_to_show[-1]
            p, t = data[s]
            return (-(p / t) if t else 0,)
        if args.sort == "total":
            return (-sum(data[s][1] for s in stages_to_show),)
        if args.sort == "functions":
            # Sort by number of distinct functions: oracle attempted is the proxy
            return (-data["oracle"][1],)
        return (cls,)

    rows = sorted(filtered.items(), key=_sort_key)

    # Column widths
    cls_w = max(len(c) for c in filtered) + 2
    stage_col_w = 26  # "passed/attempted  pct  bar"

    # Header
    header_stages = "  ".join(f"{s.upper():<{stage_col_w}}" for s in stages_to_show)
    print()
    print(f"{'CLASS':<{cls_w}}  {header_stages}")
    print("─" * (cls_w + 2 + len(stages_to_show) * (stage_col_w + 2)))

    for cls, data in rows:
        stage_cols = []
        for s in stages_to_show:
            passed, attempted = data[s]
            if attempted == 0:
                col = f"{'—':>{stage_col_w}}"
            else:
                bar = _bar(passed, attempted, width=10)
                pct = _pct(passed, attempted)
                col = f"{passed:3d}/{attempted:<3d}  {pct}  {bar}"
            stage_cols.append(f"{col:<{stage_col_w}}")
        print(f"{cls:<{cls_w}}  {'  '.join(stage_cols)}")

    # Totals row
    print("─" * (cls_w + 2 + len(stages_to_show) * (stage_col_w + 2)))
    total_cols = []
    for s in stages_to_show:
        tp = sum(data[s][0] for _, data in rows)
        ta = sum(data[s][1] for _, data in rows)
        if ta == 0:
            col = f"{'—':>{stage_col_w}}"
        else:
            bar = _bar(tp, ta, width=10)
            pct = _pct(tp, ta)
            col = f"{tp:3d}/{ta:<3d}  {pct}  {bar}"
        total_cols.append(f"{col:<{stage_col_w}}")
    print(f"{'TOTAL':<{cls_w}}  {'  '.join(total_cols)}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
