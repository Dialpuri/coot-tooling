"""CLI: exercise the agent's lookup tools by hand.

Mirrors the dispatch table in tooling.oracle.agent so what you see here is
exactly what the LLM gets back. Useful for sanity-checking the code graph
DB and for crafting prompts.

Usage:
  python -m tooling.query lookup_type gemmi::NeighborSearch
  python -m tooling.query lookup_type Residue            # ambiguous demo
  python -m tooling.query lookup_function gemmi::read_pdb_file
  python -m tooling.query list_methods gemmi::Structure
  python -m tooling.query find_symbol read_pdb_file
  python -m tooling.query find_header NeighborSearch
  python -m tooling.query search_functions populate
  python -m tooling.query get_callers coot::util::residue_atoms
  python -m tooling.query get_base_classes gemmi::NeighborSearch
  python -m tooling.query grep_codebase 'NeighborSearch\\(' --glob '*.hpp'
  python -m tooling.query read_file /opt/homebrew/include/gemmi/contact.hpp --offset 10 --limit 50
  python -m tooling.query resolve_includes --file my_snippet.cc
  python -m tooling.query inspect_pdb --chain A
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .db import connect
from .oracle.agent import (
    _tool_find_header,
    _tool_find_symbol,
    _tool_get_base_classes,
    _tool_get_callers,
    _tool_grep_codebase,
    _tool_inspect_pdb,
    _tool_list_methods,
    _tool_lookup_function,
    _tool_lookup_type,
    _tool_read_file,
    _tool_resolve_includes,
    _tool_search_functions,
)


def _add_db_subcommand(sub, name, help_text, *args):
    p = sub.add_parser(name, help=help_text)
    for a, kwargs in args:
        p.add_argument(a, **kwargs)
    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tooling.query",
        description="Run the agent's lookup tools manually against code_graph.db",
    )
    sub = parser.add_subparsers(dest="tool", required=True)

    sub.add_parser("lookup_type", help="Look up a type, listing its methods").add_argument("name")
    sub.add_parser("lookup_function", help="Look up a function by qualified name").add_argument("qualified_name")
    sub.add_parser("list_methods", help="List methods of a class").add_argument("class_name")
    sub.add_parser("get_callers", help="Show callers of a function").add_argument("qualified_name")
    sub.add_parser("find_header", help="Find which header declares a name").add_argument("name")
    sub.add_parser("search_functions", help="Substring search over function names").add_argument("name_fragment")
    sub.add_parser("get_base_classes", help="Show base classes / inheritance chain").add_argument("name")
    sub.add_parser("find_symbol", help="grep-style search across the source tree").add_argument("symbol")

    p_grep = sub.add_parser("grep_codebase", help="Grep coot + gemmi headers")
    p_grep.add_argument("pattern")
    p_grep.add_argument("--glob", default=None, help="e.g. '*.hpp'")

    p_read = sub.add_parser("read_file", help="Show a slice of a source file")
    p_read.add_argument("path")
    p_read.add_argument("--offset", type=int, default=0)
    p_read.add_argument("--limit", type=int, default=300)

    p_inc = sub.add_parser("resolve_includes", help="Suggest #includes for a code snippet")
    src = p_inc.add_mutually_exclusive_group(required=True)
    src.add_argument("--code", help="Inline snippet")
    src.add_argument("--file", type=Path, help="Read snippet from this file")

    p_pdb = sub.add_parser("inspect_pdb", help="Print contents of the standard test PDB")
    p_pdb.add_argument("--chain", default=None)

    args = parser.parse_args()

    needs_db = args.tool in {
        "lookup_type", "lookup_function", "list_methods", "get_callers",
        "find_header", "search_functions", "get_base_classes",
    }
    conn = connect() if needs_db else None
    try:
        out = _run(args, conn)
    finally:
        if conn is not None:
            conn.close()
    print(out)


def _run(args, conn) -> str:
    t = args.tool
    if t == "lookup_type":      return _tool_lookup_type(conn, args.name)
    if t == "lookup_function":  return _tool_lookup_function(conn, args.qualified_name)
    if t == "list_methods":     return _tool_list_methods(conn, args.class_name)
    if t == "get_callers":      return _tool_get_callers(conn, args.qualified_name)
    if t == "find_header":      return _tool_find_header(conn, args.name)
    if t == "search_functions": return _tool_search_functions(conn, args.name_fragment)
    if t == "get_base_classes": return _tool_get_base_classes(conn, args.name)
    if t == "find_symbol":      return _tool_find_symbol(args.symbol)
    if t == "grep_codebase":    return _tool_grep_codebase(args.pattern, args.glob)
    if t == "read_file":        return _tool_read_file(args.path, args.offset, args.limit)
    if t == "inspect_pdb":      return _tool_inspect_pdb(args.chain)
    if t == "resolve_includes":
        code = args.code if args.code is not None else args.file.read_text()
        return _tool_resolve_includes(code)
    raise SystemExit(f"unknown tool: {t}")


if __name__ == "__main__":
    main()
