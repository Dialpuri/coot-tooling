"""
Prompt rendering — assembles context from the DB into a prompt for the LLM.
"""
from __future__ import annotations

import sqlite3
import re
from pathlib import Path

from ..db import (
    get_function,
    get_containing_class,
    get_used_types,
    get_called_qnames,
    get_type,
    get_type_methods,
    get_callers_with_source,
    get_constructor_callers,
    PROJECT_ROOT,
)

# mmdb::Manager key methods are inherited from mmdb::Root / mmdb::CoorMngrRoot
# and don't appear on Manager directly, so we use a hardcoded setup pattern.
MMDB_MANAGER_SNIPPET = """\
// MMDB pointer typedef convention — every class Foo generates via DefineClass(Foo):
//   PFoo  = Foo*     PPFoo  = Foo**    RFoo  = Foo&    RPFoo  = Foo*&
// e.g. PAtom = Atom*,  PPAtom = Atom**,  PChain = Chain*,  PResidue = Residue*
//

// mmdb::Manager — load and navigate a PDB file:
//   mmdb::Manager *mol = new mmdb::Manager();
//   mol->ReadCoorFile("structure.pdb");          // load PDB
//   int nModels = mol->GetNumberOfModels();
//   mmdb::Model *model = mol->GetModel(1);        // 1-indexed
//   // selection API (alternative to manual traversal):
//   int selHnd = mol->NewSelection();
//   mol->Select(selHnd, mmdb::STYPE_RESIDUE, "//A/10", mmdb::SKEY_NEW);
//   mmdb::PPResidue selRes; int nSelRes;
//   mol->GetSelIndex(selHnd, selRes, nSelRes);
//   mol->DeleteSelection(selHnd);\
"""

# Hierarchy levels below Manager — methods are direct, so we pull from the DB.
# Each entry: (qualified_name, nav_methods_to_show)
MMDB_HIERARCHY: list[tuple[str, set[str]]] = [
    ("mmdb::Model", {
        "GetNumberOfChains", "GetChain", "GetModelNum",
    }),
    ("mmdb::Chain", {
        "GetNumberOfResidues", "GetResidue", "GetChainID",
    }),
    ("mmdb::Residue", {
        "GetNumberOfAtoms", "GetAtom",
        "GetResName", "GetSeqNum", "GetChainID", "GetInsCode",
    }),
    ("mmdb::Atom", {
        "GetAtomName", "GetElement", "GetChainID", "GetResName", "GetSeqNum",
    }),
]

_MMDB_ORDER = {"mmdb::Manager": 0}
_MMDB_ORDER.update({qname: i + 1 for i, (qname, _) in enumerate(MMDB_HIERARCHY)})

INCLUDE_ROOTS = [
    PROJECT_ROOT,
    "/opt/homebrew/Cellar/mmdb2/2.0.22/include",
    "/opt/homebrew/Cellar/clipper4coot/2.1.20180802_3/include",
    "/opt/homebrew/opt/gemmi/include",
    "/opt/homebrew/include",
]

_TEST_DATA_DIR = Path(__file__).parent.parent / "test-data"

ORACLE_INSTRUCTIONS = f"""\
Write a complete, compilable C++ program (oracle.cc) that observes the inputs
and outputs of the function marked FUNCTION TO OBSERVE below.

Requirements:
  1. Be self-contained — hardcode the test file paths below, do not use argc/argv.
       PDB: {_TEST_DATA_DIR}/example.pdb
       MTZ: {_TEST_DATA_DIR}/example.mtz
  2. Load the structure using the hardcoded path.
  3. Navigate the structure to reach a valid receiver/input for the function.
  4. Call the function.
  5. Print every input value and every meaningful output value using this format:
       INPUT  <name>: <value>
       OUTPUT <name>: <value>

Use the EXAMPLE CALLERS to understand how the function is typically invoked and
what objects are needed. Only use types and methods shown in the context below.\
"""


def _to_include(path: str) -> str:
    for root in INCLUDE_ROOTS:
        if path.startswith(root + "/"):
            return path[len(root) + 1:]
    return path


def _short_name(qname: str) -> str:
    return qname.rsplit("::", 1)[-1]


OVERRIDES_DIR = Path(__file__).parent / "overrides"


def _load_override(type_qname: str) -> str | None:
    """Return the contents of an override file for type_qname, or None.

    Files are named by replacing '::' with '__', e.g.:
      molecules_container_t       → overrides/molecules_container_t.cc
      mmdb::Residue               → overrides/mmdb__Residue.cc
    """
    stem = type_qname.replace("::", "__")
    path = OVERRIDES_DIR / f"{stem}.cc"
    if path.exists():
        return path.read_text().replace("@TEST_DATA_DIR@", str(_TEST_DATA_DIR))
    return None


def _render_type(
    conn: sqlite3.Connection,
    type_qname: str,
    summary: str,
    called_methods: set[str] | None,
    compact: bool = False,
) -> str:
    """Render a type summary with inline method comments.

    compact=True (oracle mode): omit all fields; show only constructors and
    called methods so the prompt stays focused on what the oracle needs to call.
    """
    method_rows = get_type_methods(conn, type_qname)
    comment_map = {r["display_name"]: r["comment"] or "" for r in method_rows}
    class_short  = _short_name(type_qname)

    out_lines: list[str] = []
    elided = 0

    for line in summary.splitlines():
        stripped      = line.strip()
        candidate     = stripped.rstrip(";")
        # Detect method lines by signature syntax (parentheses) rather than
        # comment_map membership — methods not yet in the functions table would
        # otherwise be misidentified as fields and dropped in compact mode.
        is_method     = "(" in candidate
        is_structural = not stripped or stripped.startswith(("class ", "struct ", "};"))
        bare_name     = candidate.split("(")[0].strip()
        is_ctor       = bare_name == class_short

        if compact and not is_method and not is_structural:
            continue  # drop field lines

        # called_methods=None → show all methods (used for return types)
        # called_methods=set() → show only constructors (used for containing class)
        if is_method and called_methods is not None and bare_name not in called_methods and not is_ctor:
            elided += 1
            continue

        comment = comment_map.get(candidate, "")
        if is_method and comment:
            out_lines.append(f"{line}  // {comment}")
        else:
            out_lines.append(line)

    if elided and out_lines and out_lines[-1].strip() == "};":
        out_lines.insert(-1, f"  // ... ({elided} more methods)")

    return "\n".join(out_lines)


def _mmdb_navigation_section(
    conn: sqlite3.Connection,
    involved_types: set[str],
    headers: dict[str, str],
) -> str | None:
    """Return a rendered MMDB hierarchy section, or None if MMDB is not involved.

    Always includes the Manager setup snippet, then renders DB-derived summaries
    for Model → Chain → Residue → Atom down to the deepest level needed.
    """
    all_mmdb = {"mmdb::Manager"} | {qname for qname, _ in MMDB_HIERARCHY}
    if not involved_types & all_mmdb:
        return None

    present = [qname for qname, _ in MMDB_HIERARCHY if qname in involved_types]
    cutoff_name = max(present, key=lambda q: _MMDB_ORDER[q]) if present else None

    lines = [MMDB_MANAGER_SNIPPET]

    for qname, nav_methods in MMDB_HIERARCHY:
        type_row = get_type(conn, qname)
        if type_row:
            inc = _to_include(type_row["file"])
            if inc not in headers:
                headers[inc] = f"MMDB hierarchy {qname}"
        lines.append(f"\n// {qname}")
        if type_row:
            lines.append(_render_type(conn, qname, type_row["summary"] or "", nav_methods, compact=True))
        if cutoff_name and qname == cutoff_name:
            break

    return "\n".join(lines)


def _extract_return_type(source_code: str, function_qname: str) -> str:
    """Parse the return type from function source, stripped of decorators."""
    fn_name = re.escape(function_qname.rsplit("::", 1)[-1])
    match = re.match(
        r'^([\w\s:<>*&,]+?)\s*\n\s*(?:[\w:]+::)+' + fn_name + r'\s*\(',
        source_code,
        re.DOTALL,
    )
    if not match:
        return ""
    raw = match.group(1)
    raw = re.sub(r'\b(const|virtual|static|inline|explicit|override)\b', '', raw)
    raw = re.sub(r'[*&]', '', raw)
    return raw.strip()


def build_oracle_prompt(conn: sqlite3.Connection, function_qname: str) -> str | None:
    fn = get_function(conn, function_qname)
    if not fn:
        return None

    # Map  type_qname -> {bare method names}  called by this function
    called_by_type: dict[str, set[str]] = {}
    for qname in get_called_qnames(conn, fn["id"]):
        if "::" in qname:
            parent, method = qname.rsplit("::", 1)
            called_by_type.setdefault(parent, set()).add(method)

    headers: dict[str, str] = {}

    # Containing class
    containing_class = None
    cls = get_containing_class(conn, function_qname)
    if cls:
        containing_class = dict(cls)
        headers[_to_include(cls["file"])] = f"containing class {cls['qualified_name']}"

    # Types used in the function body
    used_types: list[dict] = []
    for t in get_used_types(conn, fn["id"]):
        used_types.append(dict(t))
        inc = _to_include(t["file"])
        if inc not in headers:
            headers[inc] = f"{t['kind']} {t['qualified_name']}"

    # Return type — show all its methods so the oracle can inspect the output
    return_type_row = None
    ret_type_name = _extract_return_type(fn["source_code"] or "", function_qname)
    if ret_type_name:
        return_type_row = get_type(conn, ret_type_name)
        if return_type_row:
            inc = _to_include(return_type_row["file"])
            if inc not in headers:
                headers[inc] = f"return type {return_type_row['qualified_name']}"

    # Callers (example usage)
    callers = get_callers_with_source(conn, fn["id"], limit=3)

    # ---- Assemble context block ----
    ctx: list[str] = []

    ctx.append("// === INCLUDES ===")
    for inc in sorted(headers):
        ctx.append(f'#include "{inc}"')

    if containing_class:
        ctx.append(f"\n// === CONTAINING CLASS: {containing_class['qualified_name']} ===")
        ctx.append(_render_type(
            conn,
            containing_class["qualified_name"],
            containing_class["summary"] or "",
            called_by_type.get(containing_class["qualified_name"], set()),
            compact=True,
        ))

    # Containing class constructor callers — shows how to instantiate this class.
    # A hand-curated override file takes precedence over the automated DB lookup.
    if containing_class:
        cls_qname = containing_class["qualified_name"]
        override = _load_override(cls_qname)
        if override:
            ctx.append(f"\n// === {cls_qname} CONSTRUCTION (curated) ===")
            ctx.append(override.rstrip())
        else:
            ctor_callers = get_constructor_callers(conn, cls_qname)
            if ctor_callers:
                ctx.append(f"\n// === {cls_qname} CONSTRUCTOR CALLERS ===")
                for ctor_caller in ctor_callers:
                    rel = ctor_caller["file"].replace(PROJECT_ROOT + "/", "")
                    ctx.append(f"\n// {rel}")
                    if ctor_caller["comment"]:
                        ctx.append(f"// {ctor_caller['comment']}")
                    ctx.append(ctor_caller["source_code"].rstrip())

    if used_types:
        ctx.append("\n// === TYPES USED IN FUNCTION ===")
        for t in used_types:
            if containing_class and t["qualified_name"] == containing_class["qualified_name"]:
                continue
            if return_type_row and t["qualified_name"] == return_type_row["qualified_name"]:
                continue  # rendered separately below
            ctx.append(f"\n// [{t['kind']}] {t['qualified_name']}")
            ctx.append(_render_type(
                conn,
                t["qualified_name"],
                t["summary"] or "",
                called_by_type.get(t["qualified_name"], set()),
                compact=True,
            ))

    if return_type_row:
        ctx.append(f"\n// === RETURN TYPE: {return_type_row['qualified_name']} ===")
        ctx.append(_render_type(
            conn,
            return_type_row["qualified_name"],
            return_type_row["summary"] or "",
            called_methods=None,   # show all methods — these are the oracle's output accessors
            compact=True,
        ))

    # MMDB hierarchy — always show the traversal path when MMDB types are involved
    all_type_qnames: set[str] = {t["qualified_name"] for t in used_types}
    if containing_class:
        all_type_qnames.add(containing_class["qualified_name"])
    if return_type_row:
        all_type_qnames.add(return_type_row["qualified_name"])

    mmdb_section = _mmdb_navigation_section(conn, all_type_qnames, headers)
    if mmdb_section:
        ctx.append("\n// === MMDB NAVIGATION HIERARCHY ===")
        ctx.append(mmdb_section)

    if callers:
        ctx.append("\n// === EXAMPLE CALLERS ===")
        for caller in callers:
            rel = caller["file"].replace(PROJECT_ROOT + "/", "")
            ctx.append(f"\n// {rel}")
            if caller["comment"]:
                ctx.append(f"// {caller['comment']}")
            ctx.append(caller["source_code"].rstrip())

    ctx.append("\n// === FUNCTION TO OBSERVE ===")
    if fn["comment"]:
        ctx.append(f"// {fn['comment']}")
    ctx.append(fn["source_code"] or f"// (no source available) {fn['display_name']}")

    context_block = "\n".join(ctx)

    return f"{ORACLE_INSTRUCTIONS}\n\n```cpp\n{context_block}\n```\n"
