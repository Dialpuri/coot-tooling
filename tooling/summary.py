"""Print a human-readable summary of code_graph.db.

Useful for spotting indexing gaps (which subtrees are covered, which
namespaces are populated) and for ballpark sizing when deciding what to
ask the agent to port.

Usage:
  python -m tooling.summary
  python -m tooling.summary --top 20            # widen the leaderboards
  python -m tooling.summary --namespace coot    # restrict to one namespace
"""
from __future__ import annotations

import argparse
import sqlite3

from .db import connect, DB_PATH


# Namespaces we care about reporting on; everything else is bucketed as <other>.
KNOWN_NAMESPACES = ["coot", "gemmi", "mmdb", "clipper", "ProteinDB", "std", "boost"]


def _hr(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def _row(label: str, value, width: int = 32) -> None:
    print(f"  {label:<{width}} {value}")


def _top_table(rows: list[tuple], headers: tuple[str, ...]) -> None:
    if not rows:
        print("  (none)")
        return
    widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))


def _namespace_of(qname: str) -> str:
    head = qname.split("::", 1)[0]
    return head if head in KNOWN_NAMESPACES else "<other>"


def _section_overview(c: sqlite3.Connection) -> None:
    _hr("Overview")
    _row("DB path", str(DB_PATH))
    n_files = c.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    n_def   = c.execute("SELECT COUNT(*) FROM functions WHERE is_definition=1").fetchone()[0]
    n_decl  = c.execute("SELECT COUNT(*) FROM functions WHERE is_definition=0").fetchone()[0]
    n_types = c.execute("SELECT COUNT(*) FROM types").fetchone()[0]
    n_calls = c.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    n_uses  = c.execute("SELECT COUNT(*) FROM uses_type").fetchone()[0]
    _row("files",                 n_files)
    _row("functions (definitions)", n_def)
    _row("functions (declarations)", n_decl)
    _row("types",                 n_types)
    _row("call edges",            n_calls)
    _row("uses-type edges",       n_uses)


def _bucket_path(path: str) -> str:
    """Reduce a path to the most informative 2-segment bucket.

    /Users/dialpuri/lmb/coot/api/foo.cc        → coot/api
    /Users/dialpuri/lmb/coot/coot-utils/bar.cc → coot/coot-utils
    /opt/homebrew/include/gemmi/model.hpp      → include/gemmi
    /opt/homebrew/Cellar/clipper4coot/.../inc  → Cellar/clipper4coot
    """
    parts = [p for p in path.split("/") if p]
    # Find the most distinctive 2-segment slice: skip generic prefixes.
    skip_prefixes = {"Users", "dialpuri", "lmb", "opt", "homebrew"}
    keep = [p for p in parts[:-1] if p not in skip_prefixes]
    if len(keep) >= 2:
        return "/".join(keep[:2])
    return "/".join(parts[:-1]) or "/"


def _section_file_roots(c: sqlite3.Connection, top: int) -> None:
    _hr(f"File roots (top {top} subtrees)")
    counts: dict[str, int] = {}
    for (path,) in c.execute("SELECT path FROM files"):
        b = _bucket_path(path)
        counts[b] = counts.get(b, 0) + 1
    rows = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    _top_table(rows, ("root", "files"))


def _section_namespaces(c: sqlite3.Connection) -> None:
    _hr("Functions by namespace")
    rows = c.execute("SELECT qualified_name FROM functions").fetchall()
    counts: dict[str, int] = {}
    for (q,) in rows:
        ns = _namespace_of(q)
        counts[ns] = counts.get(ns, 0) + 1
    table = sorted(counts.items(), key=lambda kv: -kv[1])
    _top_table(table, ("namespace", "functions"))

    _hr("Types by namespace")
    rows = c.execute("SELECT qualified_name FROM types").fetchall()
    counts = {}
    for (q,) in rows:
        ns = _namespace_of(q)
        counts[ns] = counts.get(ns, 0) + 1
    table = sorted(counts.items(), key=lambda kv: -kv[1])
    _top_table(table, ("namespace", "types"))


def _section_kind_breakdown(c: sqlite3.Connection) -> None:
    _hr("Function kinds")
    rows = c.execute("""
        SELECT kind, COUNT(*) AS n
        FROM functions
        GROUP BY kind
        ORDER BY n DESC
    """).fetchall()
    _top_table([(r["kind"], r["n"]) for r in rows], ("kind", "count"))

    _hr("Type kinds")
    rows = c.execute("""
        SELECT kind, COUNT(*) AS n
        FROM types
        GROUP BY kind
        ORDER BY n DESC
    """).fetchall()
    _top_table([(r["kind"], r["n"]) for r in rows], ("kind", "count"))


def _section_busiest(c: sqlite3.Connection, top: int, namespace: str | None) -> None:
    where = ""
    params: tuple = ()
    if namespace:
        where = "WHERE callee_qualified_name LIKE ?"
        params = (f"{namespace}::%",)

    _hr(f"Most-called functions (top {top}{f', {namespace}::*' if namespace else ''})")
    rows = c.execute(f"""
        SELECT callee_qualified_name AS callee, COUNT(*) AS n
        FROM calls
        {where}
        GROUP BY callee
        ORDER BY n DESC
        LIMIT ?
    """, (*params, top)).fetchall()
    _top_table([(r["callee"], r["n"]) for r in rows], ("callee", "calls"))

    _hr(f"Most-referenced types (top {top}{f', {namespace}::*' if namespace else ''})")
    where = ""
    params = ()
    if namespace:
        where = "WHERE type_qualified_name LIKE ?"
        params = (f"{namespace}::%",)
    rows = c.execute(f"""
        SELECT type_qualified_name AS t, COUNT(*) AS n
        FROM uses_type
        {where}
        GROUP BY t
        ORDER BY n DESC
        LIMIT ?
    """, (*params, top)).fetchall()
    _top_table([(r["t"], r["n"]) for r in rows], ("type", "uses"))


def _section_largest_classes(c: sqlite3.Connection, top: int, namespace: str | None) -> None:
    """Classes with the most methods (definitions + declarations).

    "Class" here is the everything-before-the-last-:: of a member function's
    qualified name. Free functions in a namespace also surface here, which
    is fine — it's a rough size signal, not a strict class taxonomy.
    """
    _hr(f"Largest classes by method count (top {top})")
    counts: dict[str, int] = {}
    q = "SELECT qualified_name FROM functions WHERE qualified_name LIKE '%::%'"
    params: tuple = ()
    if namespace:
        q += " AND qualified_name LIKE ?"
        params = (f"{namespace}::%",)
    for (qn,) in c.execute(q, params):
        klass = qn.rsplit("::", 1)[0]
        counts[klass] = counts.get(klass, 0) + 1
    rows = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    _top_table(rows, ("class", "members"))


def _section_orphans(c: sqlite3.Connection) -> None:
    """Definitions that nobody calls (within the indexed code)."""
    _hr("Sanity counts")
    n_orphan = c.execute("""
        SELECT COUNT(*) FROM functions f
        WHERE f.is_definition = 1
          AND NOT EXISTS (
            SELECT 1 FROM calls c
            WHERE c.callee_qualified_name = f.qualified_name
          )
    """).fetchone()[0]
    n_total_def = c.execute("SELECT COUNT(*) FROM functions WHERE is_definition=1").fetchone()[0]
    pct = (100 * n_orphan / n_total_def) if n_total_def else 0
    _row("definitions with zero callers", f"{n_orphan} / {n_total_def}  ({pct:.1f}%)")

    n_dangling = c.execute("""
        SELECT COUNT(DISTINCT callee_qualified_name) FROM calls c
        WHERE NOT EXISTS (
            SELECT 1 FROM functions f WHERE f.qualified_name = c.callee_qualified_name
        )
    """).fetchone()[0]
    _row("call-edges to unknown callees", n_dangling)

    n_no_src = c.execute("""
        SELECT COUNT(*) FROM functions
        WHERE is_definition=1 AND (source_code IS NULL OR source_code = '')
    """).fetchone()[0]
    _row("definitions missing source_code", n_no_src)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top", type=int, default=10,
                        help="rows per leaderboard (default 10)")
    parser.add_argument("--namespace", default=None,
                        help="restrict per-namespace sections to e.g. 'coot' or 'gemmi'")
    args = parser.parse_args()

    c = connect()
    try:
        _section_overview(c)
        _section_file_roots(c, args.top)
        _section_namespaces(c)
        _section_kind_breakdown(c)
        _section_busiest(c, args.top, args.namespace)
        _section_largest_classes(c, args.top, args.namespace)
        _section_orphans(c)
    finally:
        c.close()


if __name__ == "__main__":
    main()
