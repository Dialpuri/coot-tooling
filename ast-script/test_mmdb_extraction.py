#!/usr/bin/env python3
"""Quick test: what does libclang see for MMDB method comments?"""
import sys
sys.path.insert(0, "/Users/dialpuri/lmb/coot-tooling/.venv/lib/python3.13/site-packages")

import clang.cindex
from pathlib import Path

import json, shlex
LIBCLANG     = "/opt/homebrew/opt/llvm/lib/libclang.dylib"
RESOURCE_DIR = "/opt/homebrew/opt/llvm/lib/clang/22"
MMDB_HEADER  = "/opt/homebrew/Cellar/mmdb2/2.0.22/include/mmdb2/mmdb_chain.h"
COMPILE_DB   = "/Users/dialpuri/lmb/coot-tooling/ast-data/compile_commands.json"
TARGET_FILE  = "/Users/dialpuri/lmb/coot/api/coot-molecule-change-chain-id.cc"

clang.cindex.Config.set_library_file(LIBCLANG)
_ck = clang.cindex.CursorKind

METHOD_KINDS = {_ck.CXX_METHOD, _ck.CONSTRUCTOR, _ck.DESTRUCTOR, _ck.FUNCTION_TEMPLATE}

# Use production extract_comment by importing from extract_graph
sys.path.insert(0, str(Path(__file__).parent))
from extract_graph import extract_comment as _prod_extract_comment

def extract_comment(cursor, source_lines):
    comment = _prod_extract_comment(cursor, source_lines)
    # Label the source for display purposes
    if cursor.brief_comment and cursor.brief_comment.strip():
        return ("brief", comment)
    if cursor.raw_comment and cursor.raw_comment.strip():
        return ("raw", comment)
    if source_lines and comment:
        start_line = cursor.extent.start.line
        decl_line = source_lines[start_line - 1]
        if "//" in decl_line and decl_line[decl_line.index("//") + 2:].strip() == comment:
            return ("inline", comment)
        return ("above", comment)
    return ("none", comment)


db = json.load(open(COMPILE_DB))
entry = next(e for e in db if e["file"] == TARGET_FILE)
args = shlex.split(entry["command"])[1:]
# strip flags that confuse libclang
_STRIP = {"-c", "-fcolor-diagnostics", "-fdiagnostics-color", "-fdiagnostics-color=always"}
_SKIP  = {"-o", "-MF", "-MT", "-MQ"}
cleaned, skip = [], False
for a in args:
    if skip: skip = False; continue
    if a in _SKIP: skip = True; continue
    if a in _STRIP or a == TARGET_FILE: continue
    cleaned.append(a)
cleaned += ["-resource-dir", RESOURCE_DIR]

index = clang.cindex.Index.create()
tu = index.parse(
    TARGET_FILE,
    args=cleaned,
    options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
)

with open(MMDB_HEADER) as f:
    source_lines = f.readlines()

print(f"Parsed {TARGET_FILE}, visiting for mmdb::Chain in {MMDB_HEADER}\n")

def visit(cursor, depth=0):
    loc = cursor.location
    if not loc.file or not loc.file.name.endswith("mmdb_chain.h"):
        for c in cursor.get_children():
            visit(c, depth)
        return
    if cursor.kind == _ck.CLASS_DECL and cursor.spelling == "Chain" and cursor.is_definition():
        print(f"Class: {cursor.spelling}")
        for child in cursor.get_children():
            if child.kind in METHOD_KINDS:
                kind, comment = extract_comment(child, source_lines)
                print(f"  [{kind:7}] {child.displayname[:50]:<50}  → {comment}")
        return
    for c in cursor.get_children():
        visit(c, depth + 1)

visit(tu.cursor)
