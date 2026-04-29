#!/usr/bin/env python3
"""
Layer 1: Parse all C++ files via libclang and build a code graph in SQLite.

Schema:
  files(id, path)
  functions(id, qualified_name, display_name, file_id, line_start, line_end,
            kind, is_definition, source_code)
  types(id, qualified_name, display_name, file_id, line_start, line_end,
        kind, summary)
  calls(id, caller_id, callee_qualified_name, call_line)
  uses_type(id, function_id, type_qualified_name)   -- deduplicated

Run from the repo root:
  python ast-script/extract_graph.py
"""
import json
import shlex
import sqlite3
import sys
from pathlib import Path

import clang.cindex

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LIBCLANG     = "/opt/homebrew/opt/llvm/lib/libclang.dylib"
RESOURCE_DIR = "/opt/homebrew/opt/llvm/lib/clang/22"
SYSROOT      = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
COMPILE_DB   = Path(__file__).parent.parent / "ast-data" / "compile_commands.json"
OUTPUT_DB    = Path(__file__).parent.parent / "ast-data" / "code_graph.db"
PROJECT_ROOT = "/Users/dialpuri/lmb/coot"

# Third-party include roots whose types are worth harvesting (mmdb, clipper, gemmi…)
THIRD_PARTY_ROOTS = [
    "/opt/homebrew/Cellar/mmdb2/2.0.22/include",
    "/opt/homebrew/Cellar/clipper4coot/2.1.20180802_3/include",
    "/opt/homebrew/opt/gemmi/include",
    "/opt/homebrew/Cellar/ssm/1.4_2/include",
    "/opt/homebrew/include",          # boost etc.
]

# Unexpanded shell variables in compile_commands.json — replace the whole
# bad path segment (PROJECT_ROOT/\$$var) with the real resolved prefix.
VAR_SUBS = {
    # shlex preserves the backslash from the shell escape in compile_commands.json
    f"{PROJECT_ROOT}/\\$$mmdb_prefix": "/opt/homebrew/Cellar/mmdb2/2.0.22",
}

clang.cindex.Config.set_library_file(LIBCLANG)

_ck  = clang.cindex.CursorKind
_acc = clang.cindex.AccessSpecifier

_ACCESS_NAMES = {
    _acc.PUBLIC:    "public",
    _acc.PROTECTED: "protected",
    _acc.PRIVATE:   "private",
}


def access_of(cursor: clang.cindex.Cursor) -> str:
    """Return 'public'/'protected'/'private', or '' for non-member declarations."""
    return _ACCESS_NAMES.get(cursor.access_specifier, "")


FUNCTION_KINDS = {
    _ck.FUNCTION_DECL,
    _ck.CXX_METHOD,
    _ck.CONSTRUCTOR,
    _ck.DESTRUCTOR,
    _ck.FUNCTION_TEMPLATE,
}

TYPE_KINDS = {
    _ck.CLASS_DECL,
    _ck.STRUCT_DECL,
    _ck.CLASS_TEMPLATE,
    _ck.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
    _ck.ENUM_DECL,
    _ck.TYPEDEF_DECL,
    _ck.TYPE_ALIAS_DECL,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def qualified_name(cursor: clang.cindex.Cursor) -> str:
    parts = []
    c = cursor
    while c and c.kind != _ck.TRANSLATION_UNIT:
        spelling = c.spelling or c.displayname
        if spelling:
            parts.append(spelling)
        c = c.semantic_parent
    return "::".join(reversed(parts)) if parts else (cursor.spelling or "")


_STRIP_FLAGS = {"-c", "-fcolor-diagnostics", "-fdiagnostics-color", "-fdiagnostics-color=always"}
_SKIP_NEXT   = {"-o", "-MF", "-MT", "-MQ"}


def clean_args(args: list[str], file_path: str) -> list[str]:
    """Return compiler flags safe to pass to libclang."""
    cleaned, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in _SKIP_NEXT:
            skip = True
            continue
        if a in _STRIP_FLAGS:
            continue
        if a == file_path:
            continue
        for var, val in VAR_SUBS.items():
            a = a.replace(var, val)
        if "$$" in a or "\\$$" in a:
            continue
        cleaned.append(a)
    cleaned += ["-resource-dir", RESOURCE_DIR, "-isysroot", SYSROOT]
    return cleaned


def read_lines(path: str) -> list[str]:
    try:
        with open(path, errors="replace") as f:
            return f.readlines()
    except OSError:
        return []


def slice_source(lines: list[str], start: int, end: int) -> str:
    return "".join(lines[start - 1 : end])


def extract_comment(cursor: clang.cindex.Cursor, source_lines: list[str]) -> str:
    """Return a one-line description of the function, from comments if available.

    Priority:
      1. cursor.brief_comment  — doxygen \\brief or first sentence of /** */  or ///
      2. cursor.raw_comment    — full doxygen block, cleaned to first sentence
      3. // lines immediately above the function in the source file
    """
    import re

    # 1 & 2 — structured doc comments
    brief = cursor.brief_comment
    if brief and brief.strip():
        return brief.strip()

    raw = cursor.raw_comment
    if raw and raw.strip():
        # Strip comment markers and return first non-empty sentence
        text = re.sub(r"^/\*+!?\s*|\s*\*+/$", "", raw, flags=re.DOTALL)
        text = re.sub(r"^\s*\*\s?", "", text, flags=re.MULTILINE)
        text = re.sub(r"^/{2,3}!?\s*", "", text, flags=re.MULTILINE)
        first = next((s.strip() for s in text.splitlines() if s.strip()), "")
        if first:
            return first

    # 3 — inline trailing // comment on the same line as the declaration
    if source_lines:
        start_line = cursor.extent.start.line  # 1-indexed
        decl_line = source_lines[start_line - 1]
        if "//" in decl_line:
            inline = decl_line[decl_line.index("//") + 2:].strip()
            if inline:
                return inline

    # 4 — plain // comments on consecutive lines directly above the definition
    if not source_lines:
        return ""
    start_line = cursor.extent.start.line  # 1-indexed
    comments: list[str] = []
    idx = start_line - 2  # 0-indexed line just above the function
    while idx >= 0:
        stripped = source_lines[idx].strip()
        if stripped.startswith("//"):
            comments.insert(0, stripped.lstrip("/").strip())
            idx -= 1
        else:
            break
    return " ".join(comments)


def _insert_header_methods(
    type_cursor: clang.cindex.Cursor,
    file_id: int,
    conn: sqlite3.Connection,
) -> None:
    """Insert methods of a type whose definitions live in a header file.

    Handles both third-party types (declarations, no source) and project types
    with inline definitions (captures source code and is_definition=1).
    Skips any method already present at the same (qualified_name, line_start).
    """
    method_kinds = {_ck.CXX_METHOD, _ck.CONSTRUCTOR, _ck.DESTRUCTOR, _ck.FUNCTION_TEMPLATE}
    hdr_path  = type_cursor.location.file.name if type_cursor.location.file else None
    hdr_lines = read_lines(hdr_path) if hdr_path else []

    for child in type_cursor.get_children():
        if child.kind not in method_kinds:
            continue
        qname = qualified_name(child)
        if not qname:
            continue
        ext = child.extent
        if conn.execute(
            "SELECT 1 FROM functions WHERE qualified_name = ? AND line_start = ? LIMIT 1",
            (qname, ext.start.line),
        ).fetchone():
            continue

        is_def  = int(child.is_definition())
        code    = slice_source(hdr_lines, ext.start.line, ext.end.line) if is_def and hdr_lines else ""
        comment = extract_comment(child, hdr_lines)

        conn.execute(
            """INSERT INTO functions
               (qualified_name, display_name, file_id, line_start, line_end,
                kind, is_definition, source_code, comment, access)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (qname, _method_signature(child), file_id,
             ext.start.line, ext.end.line,
             child.kind.name, is_def, code, comment, access_of(child)),
        )


def type_summary(cursor: clang.cindex.Cursor) -> str:
    """Return a concise structural summary of a type: fields + method signatures, no bodies."""
    if cursor.kind == _ck.ENUM_DECL:
        values = [
            c.spelling for c in cursor.get_children()
            if c.kind == _ck.ENUM_CONSTANT_DECL
        ]
        preview = ", ".join(values[:30])
        if len(values) > 30:
            preview += f", ... ({len(values)} total)"
        return f"enum {cursor.spelling} {{ {preview} }};"

    if cursor.kind in (_ck.TYPEDEF_DECL, _ck.TYPE_ALIAS_DECL):
        underlying = cursor.underlying_typedef_type.spelling
        return f"typedef {underlying} {cursor.spelling};"

    # CLASS_DECL, STRUCT_DECL, CLASS_TEMPLATE, partial specialisation
    is_struct = cursor.kind == _ck.STRUCT_DECL
    keyword   = "struct" if is_struct else "class"
    bases = [
        c.spelling for c in cursor.get_children()
        if c.kind == _ck.CXX_BASE_SPECIFIER
    ]
    header = f"{keyword} {cursor.displayname or cursor.spelling}"
    if bases:
        header += " : " + ", ".join(bases)

    lines = [header + " {"]
    # Default access: public for struct, private for class/class template.
    current_access: str = "public" if is_struct else "private"
    member_kinds = {
        _ck.FIELD_DECL, _ck.CXX_METHOD, _ck.CONSTRUCTOR, _ck.DESTRUCTOR,
        _ck.FUNCTION_TEMPLATE,
    }
    for child in cursor.get_children():
        if child.kind not in member_kinds:
            continue
        acc = access_of(child) or current_access
        if acc != current_access:
            lines.append(f"{acc}:")
            current_access = acc
        if child.kind == _ck.FIELD_DECL:
            lines.append(f"  {child.type.spelling} {child.spelling};")
        else:
            lines.append(f"  {_method_signature(child)};")
    lines.append("};")
    return "\n".join(lines)


def _method_signature(cursor: clang.cindex.Cursor) -> str:
    """Build a method signature with return type and parameter names.

    e.g. mmdb::Residue * GetResidue(int seqNum, const char *insCode)
    """
    params = ", ".join(
        f"{arg.type.spelling} {arg.spelling}".strip()
        for arg in cursor.get_arguments()
    )
    name        = cursor.spelling or cursor.displayname.split("(")[0]
    return_type = cursor.result_type.spelling
    if return_type and cursor.kind not in (_ck.CONSTRUCTOR, _ck.DESTRUCTOR):
        return f"{return_type} {name}({params})"
    return f"{name}({params})"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            id   INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS functions (
            id             INTEGER PRIMARY KEY,
            qualified_name TEXT    NOT NULL,
            display_name   TEXT,
            file_id        INTEGER NOT NULL REFERENCES files(id),
            line_start     INTEGER,
            line_end       INTEGER,
            kind           TEXT,
            is_definition  INTEGER,
            source_code    TEXT,
            comment        TEXT,
            access         TEXT
        );

        CREATE TABLE IF NOT EXISTS types (
            id             INTEGER PRIMARY KEY,
            qualified_name TEXT    NOT NULL UNIQUE,
            display_name   TEXT,
            file_id        INTEGER NOT NULL REFERENCES files(id),
            line_start     INTEGER,
            line_end       INTEGER,
            kind           TEXT,
            summary        TEXT
        );

        CREATE TABLE IF NOT EXISTS calls (
            id                    INTEGER PRIMARY KEY,
            caller_id             INTEGER NOT NULL REFERENCES functions(id),
            callee_qualified_name TEXT    NOT NULL,
            call_line             INTEGER
        );

        CREATE TABLE IF NOT EXISTS uses_type (
            id                    INTEGER PRIMARY KEY,
            function_id           INTEGER NOT NULL REFERENCES functions(id),
            type_qualified_name   TEXT    NOT NULL,
            UNIQUE(function_id, type_qualified_name)
        );

        CREATE INDEX IF NOT EXISTS idx_fn_qname    ON functions(qualified_name);
        CREATE INDEX IF NOT EXISTS idx_fn_file     ON functions(file_id);
        CREATE INDEX IF NOT EXISTS idx_ty_qname    ON types(qualified_name);
        CREATE INDEX IF NOT EXISTS idx_ty_file     ON types(file_id);
        CREATE INDEX IF NOT EXISTS idx_call_from   ON calls(caller_id);
        CREATE INDEX IF NOT EXISTS idx_call_to     ON calls(callee_qualified_name);
        CREATE INDEX IF NOT EXISTS idx_uses_fn     ON uses_type(function_id);
        CREATE INDEX IF NOT EXISTS idx_uses_type   ON uses_type(type_qualified_name);
    """)
    conn.commit()


def get_or_create_file(conn: sqlite3.Connection, path: str) -> int:
    row = conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO files (path) VALUES (?)", (path,))
    return cur.lastrowid


# ---------------------------------------------------------------------------
# AST traversal
# ---------------------------------------------------------------------------

def process_file(
    conn: sqlite3.Connection,
    index: clang.cindex.Index,
    entry: dict,
) -> tuple[int, int, int, int]:
    file_path = entry["file"]
    args      = shlex.split(entry["command"])
    flags     = clean_args(args[1:], file_path)

    tu = index.parse(
        file_path,
        args=flags,
        options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
    )

    source_lines = read_lines(file_path)
    file_id      = get_or_create_file(conn, file_path)

    usr_to_func_id: dict[str, int] = {}   # cursor USR -> functions.id
    n_funcs = n_calls = n_types = n_type_refs = 0

    # ------------------------------------------------------------------
    # Pass 1: collect function/method declarations and definitions.
    # Also harvests free functions defined in included project headers,
    # which never appear as entries in compile_commands.json themselves.
    # ------------------------------------------------------------------
    hdr_lines_cache: dict[str, list[str]] = {}

    def collect_functions(cursor: clang.cindex.Cursor) -> None:
        nonlocal n_funcs
        loc = cursor.location
        if not loc.file:
            for child in cursor.get_children():
                collect_functions(child)
            return

        loc_path   = loc.file.name
        in_primary = loc_path == file_path
        in_project = loc_path.startswith(PROJECT_ROOT)

        if not in_primary and not in_project:
            return  # system / third-party header — skip and don't recurse

        if cursor.kind in FUNCTION_KINDS:
            qname = qualified_name(cursor)
            ext   = cursor.extent
            is_def = cursor.is_definition()

            if in_primary:
                lines  = source_lines
                hdr_id = file_id
            else:
                if loc_path not in hdr_lines_cache:
                    hdr_lines_cache[loc_path] = read_lines(loc_path)
                lines  = hdr_lines_cache[loc_path]
                hdr_id = get_or_create_file(conn, loc_path)

            code    = slice_source(lines, ext.start.line, ext.end.line) if is_def and lines else ""
            comment = extract_comment(cursor, lines)

            existing = conn.execute(
                "SELECT id FROM functions WHERE qualified_name = ? AND line_start = ? LIMIT 1",
                (qname, ext.start.line),
            ).fetchone()
            if existing:
                func_id = existing[0]
            else:
                cur = conn.execute(
                    """INSERT INTO functions
                       (qualified_name, display_name, file_id, line_start, line_end,
                        kind, is_definition, source_code, comment, access)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (qname, _method_signature(cursor), hdr_id,
                     ext.start.line, ext.end.line,
                     cursor.kind.name, int(is_def), code, comment, access_of(cursor)),
                )
                func_id = cur.lastrowid
                n_funcs += 1

            usr = cursor.get_usr()
            if usr and (usr not in usr_to_func_id or is_def):
                usr_to_func_id[usr] = func_id

        for child in cursor.get_children():
            collect_functions(child)

    collect_functions(tu.cursor)
    conn.commit()

    # ------------------------------------------------------------------
    # Pass 2: collect type definitions from the project tree and known
    # third-party roots (mmdb, clipper, gemmi…).  UNIQUE(qualified_name)
    # + INSERT OR IGNORE silently drops duplicates across TUs.
    # ------------------------------------------------------------------
    def collect_types(cursor: clang.cindex.Cursor) -> None:
        nonlocal n_types
        loc = cursor.location
        if not loc.file:
            for child in cursor.get_children():
                collect_types(child)
            return
        loc_path = loc.file.name
        in_project = loc_path.startswith(PROJECT_ROOT)
        in_third_party = any(loc_path.startswith(r) for r in THIRD_PARTY_ROOTS)
        if not in_project and not in_third_party:
            # True system headers (libc, libc++…) — skip and don't recurse
            return

        if cursor.kind in TYPE_KINDS and cursor.is_definition():
            qname = qualified_name(cursor)
            if qname:
                ext      = cursor.extent
                summary  = type_summary(cursor)
                hdr_id   = get_or_create_file(conn, loc_path)
                conn.execute(
                    """INSERT OR IGNORE INTO types
                       (qualified_name, display_name, file_id, line_start, line_end, kind, summary)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (qname, cursor.displayname, hdr_id,
                     ext.start.line, ext.end.line,
                     cursor.kind.name, summary),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    n_types += 1

                # Capture inline methods for any type whose definition is in a
                # header — both third-party declarations and project inline defs.
                if in_third_party or (in_project and loc_path != file_path):
                    _insert_header_methods(cursor, hdr_id, conn)

        for child in cursor.get_children():
            collect_types(child)

    collect_types(tu.cursor)
    conn.commit()

    # ------------------------------------------------------------------
    # Pass 3: collect CALL_EXPR and TYPE_REF within function bodies
    # ------------------------------------------------------------------
    def collect_calls_and_refs(
        cursor: clang.cindex.Cursor, enclosing_id: int | None
    ) -> None:
        nonlocal n_calls, n_type_refs
        loc = cursor.location
        if not loc.file or loc.file.name != file_path:
            for child in cursor.get_children():
                collect_calls_and_refs(child, enclosing_id)
            return

        current_id = enclosing_id
        if cursor.kind in FUNCTION_KINDS:
            usr = cursor.get_usr()
            if usr in usr_to_func_id:
                current_id = usr_to_func_id[usr]

        if current_id is not None:
            if cursor.kind == clang.cindex.CursorKind.CALL_EXPR:
                ref = cursor.referenced
                if ref and ref.kind in FUNCTION_KINDS:
                    callee = qualified_name(ref)
                    if callee:
                        conn.execute(
                            "INSERT INTO calls (caller_id, callee_qualified_name, call_line) VALUES (?, ?, ?)",
                            (current_id, callee, loc.line),
                        )
                        n_calls += 1

            elif cursor.kind == clang.cindex.CursorKind.TYPE_REF:
                ref = cursor.referenced
                if ref and ref.kind in TYPE_KINDS:
                    tname = qualified_name(ref)
                    if tname:
                        try:
                            conn.execute(
                                "INSERT OR IGNORE INTO uses_type (function_id, type_qualified_name) VALUES (?, ?)",
                                (current_id, tname),
                            )
                            # only count actual inserts
                            if conn.execute("SELECT changes()").fetchone()[0]:
                                n_type_refs += 1
                        except sqlite3.Error:
                            pass

        for child in cursor.get_children():
            collect_calls_and_refs(child, current_id)

    collect_calls_and_refs(tu.cursor, None)
    conn.commit()

    return n_funcs, n_calls, n_types, n_type_refs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    with open(COMPILE_DB) as f:
        db = json.load(f)

    conn = sqlite3.connect(OUTPUT_DB)
    init_db(conn)

    already_done = {row[0] for row in conn.execute("SELECT path FROM files")}
    entries = [
        e for e in db
        if e["file"].startswith(PROJECT_ROOT) and e["file"] not in already_done
    ]

    total = len(entries)
    done  = len(already_done)
    print(f"Files to process: {total}  (already done: {done})")

    index      = clang.cindex.Index.create()
    tot_f = tot_c = tot_t = tot_r = 0

    for i, entry in enumerate(entries, 1):
        rel = entry["file"].removeprefix(PROJECT_ROOT + "/")
        try:
            n_f, n_c, n_t, n_r = process_file(conn, index, entry)
            tot_f += n_f; tot_c += n_c; tot_t += n_t; tot_r += n_r
            print(f"[{i:3}/{total}] {rel}  →  {n_f} funcs  {n_c} calls  {n_t} types  {n_r} type-refs")
        except Exception as exc:
            print(f"[{i:3}/{total}] ERROR {rel}: {exc}", file=sys.stderr)

    fn_defs  = conn.execute("SELECT COUNT(*) FROM functions WHERE is_definition=1").fetchone()[0]
    fn_decls = conn.execute("SELECT COUNT(*) FROM functions WHERE is_definition=0").fetchone()[0]
    n_types  = conn.execute("SELECT COUNT(*) FROM types").fetchone()[0]
    n_calls  = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    n_urefs  = conn.execute("SELECT COUNT(*) FROM uses_type").fetchone()[0]
    print(
        f"\nGraph written to {OUTPUT_DB}\n"
        f"  function definitions : {fn_defs}\n"
        f"  function declarations: {fn_decls}\n"
        f"  type definitions     : {n_types}\n"
        f"  call edges           : {n_calls}\n"
        f"  uses-type edges      : {n_urefs}"
    )
    conn.close()


if __name__ == "__main__":
    main()
