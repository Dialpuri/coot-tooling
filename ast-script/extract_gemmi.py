#!/usr/bin/env python3
"""
Index every type and free function in the installed gemmi headers.

extract_graph.py only harvests gemmi types that some coot TU transitively
pulls in — so 90+ gemmi headers (neighbor.hpp, contact.hpp, pdb.hpp,
mmread.hpp, math.hpp, unitcell.hpp, ...) never reach the DB. The gemmi-port
agent then can't look those types up.

This script fills the gap: it generates an umbrella .cpp that includes every
*.hpp in the gemmi install tree, parses it once with libclang, and inserts
gemmi types + free functions into the existing code_graph.db. Uses the
schema already created by extract_graph.py — run extract_graph.py first.

Run from the repo root:
  python ast-script/extract_gemmi.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import clang.cindex

# Reuse every helper we can from the main extractor.
from extract_graph import (
    LIBCLANG,
    RESOURCE_DIR,
    OUTPUT_DB,
    FUNCTION_KINDS,
    TYPE_KINDS,
    access_of,
    qualified_name,
    type_summary,
    _method_signature,
    read_lines,
    slice_source,
    extract_comment,
    get_or_create_file,
    init_db,
)

GEMMI_INCLUDE_ROOT = "/opt/homebrew/include"  # gemmi/*.hpp lives under here
GEMMI_HEADER_DIR   = Path(GEMMI_INCLUDE_ROOT) / "gemmi"

# libclang doesn't auto-inject the macOS SDK / libc++ paths that the clang++
# driver does, so we feed them in explicitly. Without these, any gemmi header
# that starts with `#include <map>` (ace_cc.hpp is first in alphabetical order)
# fails "file not found" and the umbrella parse bails early.
_LLVM_CXX_INCLUDE = "/opt/homebrew/opt/llvm/include/c++/v1"
# Pin the exact SDK the clang driver resolves to — the generic MacOSX.sdk
# symlink pulls in versioned headers whose macros libclang doesn't populate
# the same way (causes spurious _CTYPE_A errors).
_SDK_PATH         = "/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk"

_ck = clang.cindex.CursorKind

clang.cindex.Config.set_library_file(LIBCLANG)


def discover_headers() -> list[Path]:
    """Every *.hpp in the gemmi include tree, sorted for determinism."""
    return sorted(GEMMI_HEADER_DIR.rglob("*.hpp"))


def _probe_source(header: Path) -> str:
    """A tiny TU that includes exactly one gemmi header. Parsing headers in
    isolation keeps libclang's symbol resolution clean; the umbrella approach
    leaked degraded template signatures (e.g. `const std::string &` → `const
    int &`) depending on what came earlier in the include sequence.
    """
    rel = header.relative_to(GEMMI_INCLUDE_ROOT)
    return f"#include <{rel}>\n"


def _insert_type(
    conn: sqlite3.Connection,
    cursor: clang.cindex.Cursor,
    file_path: str,
) -> bool:
    """Insert a type definition. Returns True if newly inserted, False if
    already present (or qname empty)."""
    qname = qualified_name(cursor)
    if not qname:
        return False
    ext     = cursor.extent
    summary = type_summary(cursor)
    hdr_id  = get_or_create_file(conn, file_path)
    conn.execute(
        """INSERT OR IGNORE INTO types
           (qualified_name, display_name, file_id, line_start, line_end, kind, summary)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (qname, cursor.displayname, hdr_id,
         ext.start.line, ext.end.line,
         cursor.kind.name, summary),
    )
    return bool(conn.execute("SELECT changes()").fetchone()[0])


def _insert_function(
    conn: sqlite3.Connection,
    cursor: clang.cindex.Cursor,
    file_path: str,
    hdr_lines_cache: dict[str, list[str]],
) -> bool:
    """Insert a function declaration/definition from a header. De-duplicates
    by (qualified_name, line_start). Returns True if newly inserted."""
    qname = qualified_name(cursor)
    if not qname:
        return False
    ext = cursor.extent
    if conn.execute(
        "SELECT 1 FROM functions WHERE qualified_name = ? AND line_start = ? LIMIT 1",
        (qname, ext.start.line),
    ).fetchone():
        return False

    if file_path not in hdr_lines_cache:
        hdr_lines_cache[file_path] = read_lines(file_path)
    lines = hdr_lines_cache[file_path]

    is_def  = int(cursor.is_definition())
    code    = slice_source(lines, ext.start.line, ext.end.line) if is_def and lines else ""
    comment = extract_comment(cursor, lines)
    hdr_id  = get_or_create_file(conn, file_path)

    conn.execute(
        """INSERT INTO functions
           (qualified_name, display_name, file_id, line_start, line_end,
            kind, is_definition, source_code, comment, access)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (qname, _method_signature(cursor), hdr_id,
         ext.start.line, ext.end.line,
         cursor.kind.name, is_def, code, comment, access_of(cursor)),
    )
    return True


def _is_in_gemmi_tree(path: str | None) -> bool:
    return bool(path) and str(path).startswith(str(GEMMI_HEADER_DIR))


_FLAGS = [
    "-std=c++17",
    f"-I{GEMMI_INCLUDE_ROOT}",
    f"-isystem{_LLVM_CXX_INCLUDE}",   # libc++ (<map>, <string>, ...)
    f"-isysroot{_SDK_PATH}",          # macOS SDK (<stdlib.h>, frameworks)
    "-resource-dir", RESOURCE_DIR,
    "-Wno-everything",
]


def _visit_tu(
    conn: sqlite3.Connection,
    tu: clang.cindex.TranslationUnit,
    hdr_lines_cache: dict[str, list[str]],
) -> tuple[int, int]:
    """Harvest gemmi-tree types, methods, and free functions from one TU."""
    n_types = n_funcs = 0

    def visit(cursor: clang.cindex.Cursor) -> None:
        nonlocal n_types, n_funcs
        loc = cursor.location
        loc_path = loc.file.name if loc.file else None

        # Outside the gemmi tree we don't recurse — this excludes libc++ and
        # SDK internals and saves a lot of traversal.
        if loc_path is not None and not _is_in_gemmi_tree(loc_path):
            return

        if cursor.kind in TYPE_KINDS and cursor.is_definition() and loc_path:
            if _insert_type(conn, cursor, loc_path):
                n_types += 1
            # Capture each method declared inside this class/struct. Mirrors
            # _insert_header_methods in extract_graph.py.
            if cursor.kind in (_ck.CLASS_DECL, _ck.STRUCT_DECL,
                               _ck.CLASS_TEMPLATE,
                               _ck.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                for child in cursor.get_children():
                    child_path = (child.location.file.name
                                  if child.location.file else None)
                    if not _is_in_gemmi_tree(child_path):
                        continue
                    if child.kind in FUNCTION_KINDS:
                        if _insert_function(conn, child, child_path, hdr_lines_cache):
                            n_funcs += 1

        if cursor.kind in FUNCTION_KINDS and loc_path and _is_in_gemmi_tree(loc_path):
            parent_kind = (cursor.semantic_parent.kind
                           if cursor.semantic_parent else None)
            if parent_kind in (_ck.NAMESPACE, _ck.TRANSLATION_UNIT):
                if _insert_function(conn, cursor, loc_path, hdr_lines_cache):
                    n_funcs += 1

        for child in cursor.get_children():
            visit(child)

    visit(tu.cursor)
    return n_types, n_funcs


def collect(conn: sqlite3.Connection, headers: list[Path]) -> tuple[int, int]:
    """Parse each gemmi header in its own TU. Per-header parsing isolates
    template-symbol contamination between unrelated headers."""
    index = clang.cindex.Index.create()
    hdr_lines_cache: dict[str, list[str]] = {}
    total_types = total_funcs = 0
    parse_failures: list[tuple[str, str]] = []

    for i, header in enumerate(headers, 1):
        rel = header.relative_to(GEMMI_INCLUDE_ROOT)
        src_name = f"/tmp/__gemmi_probe_{i}.cpp"
        try:
            tu = index.parse(
                src_name,
                args=_FLAGS,
                unsaved_files=[(src_name, _probe_source(header))],
                options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
        except clang.cindex.TranslationUnitLoadError as e:
            parse_failures.append((str(rel), str(e)))
            continue

        # Per-TU error reporting — only flag errors INSIDE the gemmi header,
        # not cascading libc++ noise.
        header_errors = [
            d for d in tu.diagnostics
            if d.severity >= clang.cindex.Diagnostic.Error
            and d.location.file and _is_in_gemmi_tree(d.location.file.name)
        ]
        if header_errors:
            parse_failures.append((str(rel), header_errors[0].spelling))

        t, f = _visit_tu(conn, tu, hdr_lines_cache)
        total_types += t
        total_funcs += f
        conn.commit()

        if i % 25 == 0 or i == len(headers):
            print(f"  [{i}/{len(headers)}] +{total_types} types, +{total_funcs} funcs")

    if parse_failures:
        print(f"\n[warn] {len(parse_failures)} headers had parse issues "
              "(types/funcs in them may be missing):", file=sys.stderr)
        for name, msg in parse_failures[:10]:
            print(f"  {name}: {msg[:100]}", file=sys.stderr)

    return total_types, total_funcs


def _purge_gemmi_rows(conn: sqlite3.Connection) -> tuple[int, int]:
    """Remove every gemmi:: row before re-indexing.

    Previous runs without SDK flags captured degraded signatures (e.g.
    `read_pdb_file(const int & path)` instead of `const std::string &`) and
    the UNIQUE constraints prevent INSERT from overwriting them. A clean
    slate is the only way to pick up corrected info.
    """
    t = conn.execute(
        "DELETE FROM types WHERE qualified_name LIKE 'gemmi::%'"
    ).rowcount
    f = conn.execute(
        "DELETE FROM functions WHERE qualified_name LIKE 'gemmi::%'"
    ).rowcount
    conn.commit()
    return t, f


def main() -> None:
    if not GEMMI_HEADER_DIR.is_dir():
        print(f"gemmi headers not found at {GEMMI_HEADER_DIR}", file=sys.stderr)
        sys.exit(1)

    headers = discover_headers()
    print(f"Found {len(headers)} gemmi headers under {GEMMI_HEADER_DIR}")

    conn = sqlite3.connect(OUTPUT_DB)
    init_db(conn)  # no-op if tables exist

    purged_t, purged_f = _purge_gemmi_rows(conn)
    print(f"Purged {purged_t} stale gemmi types and {purged_f} functions")

    n_types, n_funcs = collect(conn, headers)

    after_types = conn.execute(
        "SELECT COUNT(*) FROM types WHERE qualified_name LIKE 'gemmi::%'"
    ).fetchone()[0]
    after_funcs = conn.execute(
        "SELECT COUNT(*) FROM functions WHERE qualified_name LIKE 'gemmi::%'"
    ).fetchone()[0]

    conn.close()

    print(
        f"Inserted: {n_types} types, {n_funcs} functions\n"
        f"  gemmi types    in DB: {after_types}\n"
        f"  gemmi functions in DB: {after_funcs}"
    )


if __name__ == "__main__":
    main()
