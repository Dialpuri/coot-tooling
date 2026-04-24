"""
Database access layer — thin wrappers around code_graph.db queries.
All functions accept an open sqlite3.Connection with row_factory = sqlite3.Row.
"""
import sqlite3
from pathlib import Path

DB_PATH      = Path(__file__).parent.parent / "ast-data" / "code_graph.db"
PROJECT_ROOT = "/Users/dialpuri/lmb/coot"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_function(conn: sqlite3.Connection, qname: str) -> sqlite3.Row | None:
    return conn.execute("""
        SELECT f.id, f.qualified_name, f.display_name, f.source_code, f.comment,
               fi.path AS file
        FROM functions f JOIN files fi ON fi.id = f.file_id
        WHERE f.qualified_name = ?
        ORDER BY f.is_definition DESC
        LIMIT 1
    """, (qname,)).fetchone()


def get_containing_class(conn: sqlite3.Connection, qname: str) -> sqlite3.Row | None:
    if "::" not in qname:
        return None
    parent = qname.rsplit("::", 1)[0]
    return conn.execute("""
        SELECT t.qualified_name, t.kind, t.summary, fi.path AS file
        FROM types t JOIN files fi ON fi.id = t.file_id
        WHERE t.qualified_name = ?
        LIMIT 1
    """, (parent,)).fetchone()


def get_used_types(conn: sqlite3.Connection, function_id: int) -> list[sqlite3.Row]:
    return conn.execute("""
        SELECT t.qualified_name, t.kind, t.summary, fi.path AS file
        FROM uses_type u
        JOIN types t  ON t.qualified_name = u.type_qualified_name
        JOIN files fi ON fi.id = t.file_id
        WHERE u.function_id = ?
    """, (function_id,)).fetchall()


def get_called_qnames(conn: sqlite3.Connection, function_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT callee_qualified_name FROM calls WHERE caller_id = ?",
        (function_id,)
    ).fetchall()
    return [r[0] for r in rows]


def get_type(conn: sqlite3.Connection, type_qname: str) -> sqlite3.Row | None:
    """Look up a type by exact or suffix-matched qualified name."""
    row = conn.execute("""
        SELECT t.qualified_name, t.kind, t.summary, fi.path AS file
        FROM types t JOIN files fi ON fi.id = t.file_id
        WHERE t.qualified_name = ?
        LIMIT 1
    """, (type_qname,)).fetchone()
    if row:
        return row
    # Fall back to matching on the last component (e.g. "Residue" → "mmdb::Residue")
    short = type_qname.rsplit("::", 1)[-1]
    return conn.execute("""
        SELECT t.qualified_name, t.kind, t.summary, fi.path AS file
        FROM types t JOIN files fi ON fi.id = t.file_id
        WHERE t.qualified_name LIKE ?
        LIMIT 1
    """, (f"%::{short}",)).fetchone()


def get_type_methods(conn: sqlite3.Connection, type_qname: str) -> list[sqlite3.Row]:
    return conn.execute("""
        SELECT display_name, comment
        FROM functions
        WHERE qualified_name LIKE ?
          AND kind IN ('CXX_METHOD', 'CONSTRUCTOR', 'DESTRUCTOR', 'FUNCTION_TEMPLATE')
        ORDER BY line_start
    """, (f"{type_qname}::%",)).fetchall()


def get_class_functions(
    conn: sqlite3.Connection,
    class_qname: str,
    mmdb_only: bool = False,
) -> list[str]:
    """Return qualified names of all methods in a class (definitions preferred, declarations as fallback).

    If mmdb_only is True, only return methods that use at least one mmdb:: type.
    """
    if mmdb_only:
        rows = conn.execute("""
            SELECT DISTINCT f.qualified_name
            FROM functions f
            JOIN uses_type u ON u.function_id = f.id
            WHERE f.qualified_name LIKE ?
              AND f.kind IN ('CXX_METHOD', 'CONSTRUCTOR', 'DESTRUCTOR', 'FUNCTION_TEMPLATE', 'FUNCTION_DECL')
              AND u.type_qualified_name LIKE 'mmdb::%'
            ORDER BY f.line_start
        """, (f"{class_qname}::%",)).fetchall()
    else:
        rows = conn.execute("""
            SELECT DISTINCT qualified_name
            FROM functions
            WHERE qualified_name LIKE ?
              AND kind IN ('CXX_METHOD', 'CONSTRUCTOR', 'DESTRUCTOR', 'FUNCTION_TEMPLATE', 'FUNCTION_DECL')
            ORDER BY line_start
        """, (f"{class_qname}::%",)).fetchall()
    return [r[0] for r in rows]


def get_constructor_callers(
    conn: sqlite3.Connection,
    type_qname: str,
    limit: int = 5,
) -> list[sqlite3.Row]:
    """Return callers of type_qname's constructor, shortest source first."""
    short = type_qname.rsplit("::", 1)[-1]
    ctor_qname = f"{type_qname}::{short}"
    return conn.execute("""
        SELECT DISTINCT f.qualified_name, f.display_name, f.source_code, f.comment,
               fi.path AS file
        FROM calls c
        JOIN functions f  ON f.id = c.caller_id
        JOIN files fi     ON fi.id = f.file_id
        WHERE c.callee_qualified_name = ?
          AND f.is_definition = 1
          AND f.source_code IS NOT NULL
          AND f.source_code != ''
        ORDER BY LENGTH(f.source_code) ASC
        LIMIT ?
    """, (ctor_qname, limit)).fetchall()


def get_internal_call_deps(
    conn: sqlite3.Connection, qnames: list[str],
) -> dict[str, set[str]]:
    """For every qname in the batch, return the subset of qnames it calls
    that are ALSO in the batch. Self-calls (direct recursion) are ignored.

    Result shape: {caller_qname: {callee_qname, ...}}.
    Every input qname is present as a key, possibly with an empty set.
    """
    if not qnames:
        return {}
    placeholders = ",".join("?" * len(qnames))
    rows = conn.execute(f"""
        SELECT DISTINCT f.qualified_name, c.callee_qualified_name
        FROM calls c
        JOIN functions f ON f.id = c.caller_id
        WHERE f.qualified_name IN ({placeholders})
          AND c.callee_qualified_name IN ({placeholders})
    """, (*qnames, *qnames)).fetchall()
    deps: dict[str, set[str]] = {q: set() for q in qnames}
    for caller, callee in rows:
        if caller != callee:
            deps[caller].add(callee)
    return deps


def get_callers_with_source(
    conn: sqlite3.Connection,
    function_id: int,
    limit: int = 2,
) -> list[sqlite3.Row]:
    """Return callers that have source code, shortest first (easier to read)."""
    return conn.execute("""
        SELECT DISTINCT f.qualified_name, f.display_name, f.source_code, f.comment,
               fi.path AS file
        FROM calls c
        JOIN functions f  ON f.id = c.caller_id
        JOIN files fi     ON fi.id = f.file_id
        WHERE c.callee_qualified_name = (
            SELECT qualified_name FROM functions WHERE id = ?
        )
          AND f.is_definition = 1
          AND f.source_code IS NOT NULL
          AND f.source_code != ''
        ORDER BY LENGTH(f.source_code) ASC
        LIMIT ?
    """, (function_id, limit)).fetchall()
