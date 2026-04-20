#!/usr/bin/env python3
"""
Layer 3: Query the code graph.

Given a natural-language question or a function name, retrieves the most
relevant functions via semantic search, expands the result set by walking
the call graph and type graph, then assembles a compact context document
ready to paste into a reasoning model (or pipe to one via API).

Usage:
  conda activate faiss
  python ast-script/query.py "how does ligand fitting work"
  python ast-script/query.py "coot::molecule_t::get_bonds_mesh" --hops 2
  python ast-script/query.py "electron density map" --top 5 --hops 1 --no-types

  # Resolve headers needed to write a test for a specific function:
  python ast-script/query.py --headers "coot::molecule_t::get_bonds_mesh"
"""
import argparse
import json
import os
import sqlite3
from pathlib import Path

# Prevent sentence-transformers from phoning home to check for model updates.
# The model is already cached locally; no network access is needed.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH    = Path(__file__).parent.parent / "ast-data" / "code_graph.db"
INDEX_PATH = Path(__file__).parent.parent / "ast-data" / "index.faiss"
META_PATH  = Path(__file__).parent.parent / "ast-data" / "index_meta.json"
PROJECT_ROOT = "/Users/dialpuri/lmb/coot"

MODEL_NAME = "google/embeddinggemma-300m"



# ---------------------------------------------------------------------------
# Load (cached globals so interactive use is fast)
# ---------------------------------------------------------------------------

_model  = None
_index  = None
_meta   = None
_conn   = None


def _load():
    global _model, _index, _meta, _conn
    if _model is None:
        print("Loading model...", flush=True)
        _model = SentenceTransformer(MODEL_NAME)
    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))
        _meta  = json.loads(META_PATH.read_text())
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH)
        _conn.row_factory = sqlite3.Row
    return _model, _index, _meta, _conn


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _get_function(conn, function_id: int) -> sqlite3.Row | None:
    return conn.execute("""
        SELECT f.id, f.qualified_name, f.display_name, f.kind,
               f.line_start, f.line_end, f.source_code, f.comment,
               fi.path AS file
        FROM functions f JOIN files fi ON fi.id = f.file_id
        WHERE f.id = ?
    """, (function_id,)).fetchone()


def _callees(conn, function_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT callee_qualified_name FROM calls WHERE caller_id = ?",
        (function_id,)
    ).fetchall()
    return [r[0] for r in rows]


def _callers(conn, function_id: int) -> list[str]:
    rows = conn.execute("""
        SELECT DISTINCT f.qualified_name
        FROM calls c JOIN functions f ON f.id = c.caller_id
        WHERE c.callee_qualified_name = (
            SELECT qualified_name FROM functions WHERE id = ?
        )
    """, (function_id,)).fetchall()
    return [r[0] for r in rows]


def _used_types(conn, function_id: int) -> list[sqlite3.Row]:
    return conn.execute("""
        SELECT t.qualified_name, t.kind, t.summary
        FROM uses_type u JOIN types t ON t.qualified_name = u.type_qualified_name
        WHERE u.function_id = ?
    """, (function_id,)).fetchall()


def _annotated_type_summary(conn, type_qname: str, raw_summary: str) -> str:
    """Append a method-descriptions block to a type summary (used in semantic context)."""
    methods = conn.execute("""
        SELECT display_name, comment
        FROM functions
        WHERE qualified_name LIKE ?
          AND kind IN ('CXX_METHOD','CONSTRUCTOR','DESTRUCTOR','FUNCTION_TEMPLATE')
          AND comment IS NOT NULL AND comment != ''
        ORDER BY line_start
    """, (f"{type_qname}::%",)).fetchall()

    if not methods:
        return raw_summary

    descriptions = "\n".join(
        f"//   {m['display_name']}: {m['comment']}" for m in methods
    )
    return f"{raw_summary.rstrip()}\n\n// Method descriptions:\n{descriptions}"


def _class_for_prompt(
    conn,
    type_qname: str,
    raw_summary: str,
    called_methods: set[str] | None = None,
) -> str:
    """Render a class for --prompt output.

    If called_methods is provided (a set of display_names actually called by
    the function under test), only those method lines are kept — everything
    else is replaced with a single '// ... (N more methods)' elision.
    Comments are added inline where available.
    """
    rows = conn.execute("""
        SELECT display_name, comment
        FROM functions
        WHERE qualified_name LIKE ?
          AND kind IN ('CXX_METHOD','CONSTRUCTOR','DESTRUCTOR','FUNCTION_TEMPLATE')
        ORDER BY line_start
    """, (f"{type_qname}::%",)).fetchall()

    comment_map = {r["display_name"]: r["comment"] or "" for r in rows}

    lines = []
    elided = 0
    for line in raw_summary.splitlines():
        candidate = line.strip().rstrip(";")   # display_name e.g. "GetResidue(int)"
        is_method = candidate in comment_map
        bare_name = candidate.split("(")[0]    # strip params  e.g. "GetResidue"

        if is_method and called_methods is not None and bare_name not in called_methods:
            elided += 1
            continue

        if is_method:
            comment = comment_map.get(candidate, "")
            lines.append(f"{line}   // {comment}" if comment else line)
        else:
            lines.append(line)

    # Insert elision notice before closing brace
    if elided and lines and lines[-1].strip() == "};":
        lines.insert(-1, f"  // ... ({elided} more methods not used here)")

    return "\n".join(lines)


def _function_by_name(conn, qname: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id FROM functions WHERE qualified_name = ? AND is_definition = 1 LIMIT 1",
        (qname,)
    ).fetchone()


def _containing_class(conn, qualified_name: str) -> sqlite3.Row | None:
    """Return the types row for the class this function belongs to, if any."""
    # Strip the final ::component to get the parent qualified name
    if "::" not in qualified_name:
        return None
    parent = qualified_name.rsplit("::", 1)[0]
    return conn.execute(
        "SELECT qualified_name, kind, file_id FROM types WHERE qualified_name = ? LIMIT 1",
        (parent,)
    ).fetchone()


# Known third-party include roots — used to make paths relative for #include
_INCLUDE_ROOTS = [
    PROJECT_ROOT,
    "/opt/homebrew/Cellar/mmdb2/2.0.22/include",
    "/opt/homebrew/Cellar/clipper4coot/2.1.20180802_3/include",
    "/opt/homebrew/opt/gemmi/include",
    "/opt/homebrew/include",
]


def _to_include_path(abs_path: str) -> str:
    """Convert an absolute header path to the shortest includable relative path."""
    for root in _INCLUDE_ROOTS:
        if abs_path.startswith(root + "/"):
            return abs_path[len(root) + 1:]
    return abs_path


def _gather_test_context(function_qname: str) -> dict:
    """
    Collect everything needed to write a test for function_qname:
      - the function's source code
      - its containing class summary
      - summaries of every type used in the body
      - resolved #include paths for all of the above
    Returns a dict with keys: function, containing_class, used_types, headers.
    """
    _, _, _, conn = _load()

    fn_row = conn.execute("""
        SELECT f.id, f.qualified_name, f.display_name, f.source_code, f.comment,
               fi.path AS file
        FROM functions f JOIN files fi ON fi.id = f.file_id
        WHERE f.qualified_name = ?
        ORDER BY f.is_definition DESC
        LIMIT 1
    """, (function_qname,)).fetchone()

    if not fn_row:
        return {}

    function_id = fn_row["id"]
    headers: dict[str, str] = {}   # include_path -> reason
    used_types: list[dict]  = []

    # Containing class
    containing_class = None
    cls = _containing_class(conn, function_qname)
    if cls:
        cls_full = conn.execute("""
            SELECT t.qualified_name, t.kind, t.summary, fi.path AS file
            FROM types t JOIN files fi ON fi.id = t.file_id
            WHERE t.qualified_name = ?
        """, (cls["qualified_name"],)).fetchone()
        if cls_full:
            containing_class = dict(cls_full)
            inc = _to_include_path(cls_full["file"])
            headers[inc] = f"containing class {cls_full['qualified_name']}"

    # Types used in the function body
    used_rows = conn.execute("""
        SELECT t.qualified_name, t.kind, t.summary, fi.path AS file
        FROM uses_type u
        JOIN types t  ON t.qualified_name = u.type_qualified_name
        JOIN files fi ON fi.id = t.file_id
        WHERE u.function_id = ?
    """, (function_id,)).fetchall()

    for t in used_rows:
        used_types.append(dict(t))
        inc = _to_include_path(t["file"])
        if inc not in headers:
            headers[inc] = f"{t['kind']} {t['qualified_name']}"

    return {
        "function":         dict(fn_row),
        "containing_class": containing_class,
        "used_types":       used_types,
        "headers":          headers,
    }


def resolve_headers(function_qname: str) -> None:
    """Print #include lines needed to test the given function."""
    ctx = _gather_test_context(function_qname)
    if not ctx:
        print(f"Function not found: {function_qname}")
        return

    print(f"\n// Headers for: {function_qname}\n")
    for inc, reason in sorted(ctx["headers"].items()):
        print(f'#include "{inc}"'  + f"   // {reason}")
    print(f"\n// {len(ctx['headers'])} header(s) total")


def build_test_prompt(function_qname: str) -> None:
    """Print a self-contained prompt block for asking a model to write a test."""
    ctx = _gather_test_context(function_qname)
    if not ctx:
        print(f"Function not found: {function_qname}")
        return

    fn   = ctx["function"]
    conn = _conn
    lines = []

    # Collect which methods of each type are actually called by this function.
    # Use callee_qualified_name directly — no JOIN — so third-party methods
    # (mmdb, clipper etc.) that aren't in our functions table are included.
    called_rows = conn.execute(
        "SELECT callee_qualified_name FROM calls WHERE caller_id = ?",
        (fn["id"],)
    ).fetchall()

    # Map  type_qname -> {bare_method_name, ...}  e.g. "mmdb::Chain" -> {"GetNumberOfResidues", "GetResidue"}
    # We store bare names (no params) so we can prefix-match against display_name lines.
    called_by_type: dict[str, set[str]] = {}
    for r in called_rows:
        qname = r["callee_qualified_name"]
        if "::" in qname:
            parent, method = qname.rsplit("::", 1)
            called_by_type.setdefault(parent, set()).add(method)

    # --- Headers ---
    lines.append("// === INCLUDES ===")
    for inc in sorted(ctx["headers"]):
        lines.append(f'#include "{inc}"')

    # --- Containing class ---
    if ctx["containing_class"]:
        cls = ctx["containing_class"]
        lines.append(f"\n// === CONTAINING CLASS: {cls['qualified_name']} ===")
        called = called_by_type.get(cls["qualified_name"])
        lines.append(_class_for_prompt(conn, cls["qualified_name"], cls["summary"] or "", called))

    # --- Used types ---
    if ctx["used_types"]:
        lines.append("\n// === TYPES USED IN FUNCTION ===")
        for t in ctx["used_types"]:
            if ctx["containing_class"] and t["qualified_name"] == ctx["containing_class"]["qualified_name"]:
                continue
            lines.append(f"\n// [{t['kind']}] {t['qualified_name']}")
            called = called_by_type.get(t["qualified_name"])
            lines.append(_class_for_prompt(conn, t["qualified_name"], t["summary"] or "", called))

    # --- Function source ---
    lines.append(f"\n// === FUNCTION TO TEST ===")
    if fn.get("comment"):
        lines.append(f"// {fn['comment']}")
    lines.append(fn["source_code"] or f"// (no source) {fn['display_name']}")

    # --- Instruction ---
    lines.append(f"""
// === TASK ===
// Write a C++ unit test for the function above.
// Use the class definitions and includes provided.
// Do not invent methods or fields that are not shown.""")

    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Graph expansion
# ---------------------------------------------------------------------------

def expand(
    conn,
    seed_ids: list[int],
    hops: int,
    include_types: bool,
) -> tuple[list[dict], list[dict]]:
    """
    BFS over the call graph from seed_ids up to `hops` steps.
    Returns (function_records, type_records) deduplicated.
    """
    visited_fns:   set[int]  = set()
    visited_types: set[str]  = set()
    fn_records:    list[dict] = []
    type_records:  list[dict] = []

    frontier = list(seed_ids)
    for _ in range(hops + 1):
        next_frontier = []
        for fid in frontier:
            if fid in visited_fns:
                continue
            visited_fns.add(fid)
            row = _get_function(conn, fid)
            if not row:
                continue

            callees = _callees(conn, fid)
            callers = _callers(conn, fid)

            fn_records.append({
                "id":             fid,
                "qualified_name": row["qualified_name"],
                "display_name":   row["display_name"],
                "file":           row["file"].replace(PROJECT_ROOT + "/", ""),
                "line_start":     row["line_start"],
                "line_end":       row["line_end"],
                "source_code":    row["source_code"] or "",
                "comment":        row["comment"] or "",
                "callees":        callees,
                "callers":        callers,
            })

            if include_types:
                for t in _used_types(conn, fid):
                    if t["qualified_name"] not in visited_types:
                        visited_types.add(t["qualified_name"])
                        type_records.append({
                            "qualified_name": t["qualified_name"],
                            "kind":           t["kind"],
                            "summary":        t["summary"],
                        })

            # Expand to direct callees (only project functions in the DB)
            for callee_name in callees:
                row2 = _function_by_name(conn, callee_name)
                if row2:
                    next_frontier.append(row2["id"])

        frontier = next_frontier

    return fn_records, type_records


# ---------------------------------------------------------------------------
# Context document rendering
# ---------------------------------------------------------------------------

def render_context(fn_records: list[dict], type_records: list[dict]) -> str:
    lines = []

    if type_records:
        lines.append("=" * 60)
        lines.append("TYPES USED")
        lines.append("=" * 60)
        for t in type_records:
            lines.append(f"\n// [{t['kind']}] {t['qualified_name']}")
            summary = _annotated_type_summary(_conn, t["qualified_name"], t["summary"] or "")
            lines.append(summary)

    lines.append("\n" + "=" * 60)
    lines.append("FUNCTIONS")
    lines.append("=" * 60)
    for fn in fn_records:
        lines.append(f"\n// {fn['file']}:{fn['line_start']}")
        if fn.get("comment"):
            lines.append(f"// {fn['comment']}")
        if fn["callers"]:
            lines.append(f"// Called by: {', '.join(fn['callers'][:5])}")
        if fn["callees"]:
            lines.append(f"// Calls:     {', '.join(fn['callees'][:10])}")
        if fn["source_code"]:
            lines.append(fn["source_code"].rstrip())
        else:
            lines.append(f"// (declaration only) {fn['display_name']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main query entry point
# ---------------------------------------------------------------------------

def query(
    question: str,
    top_k: int = 5,
    hops: int = 1,
    include_types: bool = True,
) -> str:
    model, index, meta, conn = _load()

    vec = model.encode([question], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(vec, top_k)

    print(f"\nTop {top_k} semantic matches:")
    seed_ids = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        if idx == -1:
            continue
        m = meta[idx]
        rel = m["file"].replace(PROJECT_ROOT + "/", "")
        print(f"  {rank}. [{score:.3f}] {m['qualified_name']}  ({rel}:{m['line_start']})")
        seed_ids.append(m["function_id"])

    # print(f"\nExpanding {hops} hop(s) over call graph...")
    # fn_records, type_records = expand(conn, seed_ids, hops, include_types)
    # print(f"Context: {len(fn_records)} functions, {len(type_records)} types")
    #
    # return render_context(fn_records, type_records)
    return ""

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Query the Coot code graph")
    parser.add_argument("question", help="Natural-language question or fully-qualified function name")
    parser.add_argument("--top",      type=int,  default=5,    help="Semantic matches to seed from (default 5)")
    parser.add_argument("--hops",     type=int,  default=1,    help="Call-graph expansion hops (default 1)")
    parser.add_argument("--no-types", action="store_true",     help="Omit type summaries from context")
    parser.add_argument("--out",      type=str,  default=None, help="Write context to file instead of stdout")
    parser.add_argument("--headers",  action="store_true",
                        help="Resolve #include headers needed to test the given function (exact qualified name required)")
    parser.add_argument("--prompt",   action="store_true",
                        help="Build a full test-writing prompt: headers + class definitions + function source")
    args = parser.parse_args()

    if args.headers:
        resolve_headers(args.question)
        return

    if args.prompt:
        build_test_prompt(args.question)
        return

    context = query(args.question, args.top, args.hops, not args.no_types)

    if args.out:
        Path(args.out).write_text(context)
        print(f"\nContext written to {args.out}")
    else:
        print("\n" + context)


if __name__ == "__main__":
    main()
