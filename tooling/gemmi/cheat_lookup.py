"""Lookup helpers backed by curated artifacts:

* `mmdb_to_gemmi(method)` reads `tooling/gemmi/.cheat_cache/*.json` (built by
  `build_cheat_sheet.py`) and returns the gemmi equivalent for an MMDB method.
  Surfaces the curated NO_EQUIVALENT explanations from the same cache.

* `include_for_symbol(symbol)` returns the `#include` directive that defines
  a gemmi (or gtest) symbol. Built lazily by scanning the gemmi header tree
  for top-level declarations once per session and cached on disk so subsequent
  agent runs skip the scan.

Both are stateless from the LLM's perspective — call by name and get an
authoritative answer instead of grep'ing for the same thing every run.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from ..oracle.compile import GEMMI_INCLUDE
from ..test.compile import GTEST_INCLUDE

CACHE_DIR = Path(__file__).parent / ".cheat_cache"
INDEX_PATH = Path(__file__).parent / ".symbol_index.json"


# ── MMDB → gemmi method lookup ────────────────────────────────────────────────

def _normalise_mmdb_query(query: str) -> str:
    """Accept several spellings and return the canonical qualified name.

    Examples:
      'GetSeqNum'                       → 'GetSeqNum'  (returned as-is)
      'mmdb::Residue::GetSeqNum'        → 'mmdb::Residue::GetSeqNum'
      'residue->GetSeqNum'              → 'GetSeqNum'
      'residue.GetSeqNum'               → 'GetSeqNum'
    """
    q = query.strip()
    if "->" in q:
        q = q.split("->", 1)[-1]
    if q.startswith("."):
        q = q[1:]
    # Drop trailing parens / args
    q = re.sub(r"\s*\(.*", "", q).strip()
    return q


# Lines like:
#   '  residue->GetSeqNum()            → residue.seqid.num.value // some note'
# in the inline GEMMI_CHEAT_SHEET docstring. We parse them at load time so the
# hand-curated mappings are accessible via the same lookup tool as the auto-
# generated .cheat_cache entries.
_INLINE_RE = re.compile(
    r"^\s*(?P<recv>\w+)->(?P<method>\w+)\s*\([^)]*\)\s+→\s+"
    r"(?P<gemmi>.+?)\s*(?://\s*(?P<note>.*))?$"
)
_RECV_TO_CLASS = {
    "mol":     "mmdb::Manager",
    "model":   "mmdb::Model",
    "chain":   "mmdb::Chain",
    "residue": "mmdb::Residue",
    "atom":    "mmdb::Atom",
}


def _load_inline_mappings() -> list[dict]:
    """Parse the prose accessor table inside agent.py's GEMMI_CHEAT_SHEET."""
    try:
        from .agent import GEMMI_CHEAT_SHEET
    except ImportError:
        return []
    entries: list[dict] = []
    for line in GEMMI_CHEAT_SHEET.splitlines():
        m = _INLINE_RE.match(line)
        if not m:
            continue
        recv = m.group("recv")
        method = m.group("method")
        cls = _RECV_TO_CLASS.get(recv)
        if not cls:
            continue
        entries.append({
            "qualified_name": f"{cls}::{method}",
            "gemmi":          m.group("gemmi").strip(),
            "note":           (m.group("note") or "").strip(),
            "skipped":        False,
        })
    return entries


@lru_cache(maxsize=1)
def _load_cheat_cache() -> dict[str, dict]:
    """{tail-name → entry} and {qualified-name → entry}, merged.

    Sources, in priority order (first wins on key collision):
      1. Hand-curated inline mappings parsed from `GEMMI_CHEAT_SHEET`
      2. Auto-generated `.cheat_cache/*.json` files

    Tail names (e.g. 'GetSeqNum') let the agent ask without the namespace,
    which is the common case in actual port code.
    """
    out: dict[str, dict] = {}

    # Inline first — they're the hand-curated source of truth.
    for entry in _load_inline_mappings():
        qn = entry["qualified_name"]
        out[qn] = entry
        tail = qn.rsplit("::", 1)[-1]
        out.setdefault(tail, entry)

    if not CACHE_DIR.exists():
        return out
    for p in sorted(CACHE_DIR.glob("*.json")):
        try:
            entry = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        qn = entry.get("qualified_name") or ""
        if not qn:
            continue
        out.setdefault(qn, entry)
        tail = qn.rsplit("::", 1)[-1]
        out.setdefault(tail, entry)
    return out


def mmdb_to_gemmi(method: str) -> str:
    """Look up a curated MMDB → gemmi mapping.

    Returns a multi-line string suitable for direct inclusion in a tool
    response. Falls back to a "not in catalog — use grep_codebase" message
    when the method isn't covered.
    """
    cache = _load_cheat_cache()
    q = _normalise_mmdb_query(method)
    entry = cache.get(q)
    if entry is None:
        # Fuzzy fallback: try a substring match on tail names.
        tail_matches = [
            (qn, e) for qn, e in cache.items()
            if "::" in qn and q.lower() in qn.rsplit("::", 1)[-1].lower()
        ]
        if not tail_matches:
            return (
                f"No curated mapping for '{method}'. "
                "Try grep_codebase to search the gemmi headers, or "
                "lookup_type for the receiver class to see equivalent methods."
            )
        if len(tail_matches) > 1:
            shown = "\n".join(f"  {qn}" for qn, _ in tail_matches[:8])
            return (
                f"'{method}' is ambiguous — multiple matches in the curated "
                f"cache. Re-call with a fully-qualified name:\n{shown}"
            )
        entry = tail_matches[0][1]

    qn = entry.get("qualified_name", "")
    note = entry.get("note", "")
    gemmi = entry.get("gemmi", "")
    skipped = entry.get("skipped", False)

    if skipped:
        return (
            f"{qn}: NO direct gemmi equivalent.\n"
            f"  Why: {note or '(no note)'}\n"
            "  Strategy: implement the behaviour manually using the gemmi "
            "primitives (see lookup_type / grep_codebase)."
        )

    out = [f"{qn} → {gemmi}"]
    if note:
        out.append(f"  // {note}")
    return "\n".join(out)


# ── symbol → header index ─────────────────────────────────────────────────────

# Patterns that introduce top-level declarations whose name we want to capture.
# We deliberately keep these conservative — false positives would point the
# agent at the wrong header.
_DECL_PATTERNS = [
    # `inline X foo(`, `GEMMI_DLL X foo(`, `static inline X foo(`
    re.compile(r"^(?:GEMMI_DLL\s+|inline\s+|static\s+|template\s*<[^>]*>\s*)*"
               r"(?:[\w:<>,\s\*&]+?)\s+(\w+)\s*\("),
    # `struct Name`, `class Name`, `enum Name` — strip trailing `:` / `{`.
    re.compile(r"^(?:struct|class|enum(?:\s+class)?)\s+(\w+)\b"),
    # `using NAME = ...` and `typedef ... NAME;`
    re.compile(r"^using\s+(\w+)\s*="),
    re.compile(r"^typedef\s+.+?\b(\w+)\s*;"),
]

# Symbols here are known to live in this header but might be missed by the
# pattern scan (e.g. macros, declared inside an extern "C" block, etc.).
_SEED_SYMBOLS: dict[str, str] = {
    # gtest macros
    "TEST":              "<gtest/gtest.h>",
    "TEST_F":            "<gtest/gtest.h>",
    "TEST_P":            "<gtest/gtest.h>",
    "EXPECT_EQ":         "<gtest/gtest.h>",
    "EXPECT_NE":         "<gtest/gtest.h>",
    "EXPECT_TRUE":       "<gtest/gtest.h>",
    "EXPECT_FALSE":      "<gtest/gtest.h>",
    "EXPECT_FLOAT_EQ":   "<gtest/gtest.h>",
    "EXPECT_DOUBLE_EQ":  "<gtest/gtest.h>",
    "EXPECT_NEAR":       "<gtest/gtest.h>",
    "EXPECT_LT":         "<gtest/gtest.h>",
    "EXPECT_LE":         "<gtest/gtest.h>",
    "EXPECT_GT":         "<gtest/gtest.h>",
    "EXPECT_GE":         "<gtest/gtest.h>",
    "ASSERT_EQ":         "<gtest/gtest.h>",
    "ASSERT_TRUE":       "<gtest/gtest.h>",
    "RUN_ALL_TESTS":     "<gtest/gtest.h>",
    # Common gemmi types that may not have a 1-line decl
    "Structure":         "<gemmi/model.hpp>",
    "Model":             "<gemmi/model.hpp>",
    "Chain":             "<gemmi/model.hpp>",
    "Residue":           "<gemmi/model.hpp>",
    "Atom":              "<gemmi/model.hpp>",
    "CRA":               "<gemmi/model.hpp>",
    "ResidueId":         "<gemmi/model.hpp>",
    "SeqId":             "<gemmi/model.hpp>",
    "EntityType":        "<gemmi/model.hpp>",
    "PolymerType":       "<gemmi/model.hpp>",
    "Element":           "<gemmi/elem.hpp>",
    "Vec3":              "<gemmi/math.hpp>",
    "Position":          "<gemmi/unitcell.hpp>",
    "Fractional":        "<gemmi/unitcell.hpp>",
    "UnitCell":          "<gemmi/unitcell.hpp>",
    "Mat33":             "<gemmi/math.hpp>",
    "Transform":         "<gemmi/math.hpp>",
    "NeighborSearch":    "<gemmi/neighbor.hpp>",
    "ContactSearch":     "<gemmi/contact.hpp>",
    "Mtz":               "<gemmi/mtz.hpp>",
    "Grid":              "<gemmi/grid.hpp>",
    "Ccp4":              "<gemmi/ccp4.hpp>",
    "ChemComp":          "<gemmi/chemcomp.hpp>",
    "MonLib":            "<gemmi/monlib.hpp>",
    "Topo":              "<gemmi/topo.hpp>",
    # Common functions that the agent invents the wrong path for
    "read_pdb_file":     "<gemmi/pdb.hpp>",
    "read_structure":    "<gemmi/mmread.hpp>",
    "read_ccp4_map":     "<gemmi/ccp4.hpp>",
    "read_mtz_file":     "<gemmi/mtz.hpp>",
    "setup_entities":    "<gemmi/polyheur.hpp>",
    "remove_waters":     "<gemmi/polyheur.hpp>",
    "remove_alternative_conformations": "<gemmi/modify.hpp>",
    "remove_empty_children":            "<gemmi/modify.hpp>",
    "remove_ligands_and_waters":        "<gemmi/polyheur.hpp>",
    "transform_pos_and_adp":            "<gemmi/modify.hpp>",
    "calculate_center_of_mass":         "<gemmi/calculate.hpp>",
    "find_tabulated_residue":           "<gemmi/resinfo.hpp>",
    "make_assembly":     "<gemmi/assembly.hpp>",
    "write_pdb":         "<gemmi/to_pdb.hpp>",
    "make_mmcif_block":  "<gemmi/to_mmcif.hpp>",
    "update_mmcif_block":               "<gemmi/to_mmcif.hpp>",
}


def _scan_header(path: Path, gemmi_root: Path, gtest_root: Path) -> list[tuple[str, str]]:
    """Scan one header file for top-level declarations.

    Returns [(symbol, include-string), ...]. Skips lines that look like
    forward declarations of templates we'd misclassify.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []

    if path.is_relative_to(gemmi_root):
        rel = path.relative_to(gemmi_root.parent)  # 'gemmi/foo.hpp'
        include = f"<{rel}>"
    elif path.is_relative_to(gtest_root):
        rel = path.relative_to(gtest_root)
        include = f"<{rel}>"
    else:
        return []

    out: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        # Skip preprocessor and comments — not perfect but cheap.
        if not line or line.startswith(("//", "*", "/*", "#")):
            continue
        # Track only top-level (file-scope) and gemmi-namespace declarations.
        # Anything inside a class body is indented in this codebase and won't
        # match `^pattern` after .strip().
        for rx in _DECL_PATTERNS:
            m = rx.match(line)
            if m:
                name = m.group(1)
                # Filter out C++ keywords / non-symbol tokens that occasionally
                # get matched by the patterns above.
                if name and name[0].isalpha() and name not in (
                    "if", "for", "while", "do", "switch", "return", "namespace",
                    "operator", "explicit", "virtual", "const", "static",
                ):
                    out.append((name, include))
                break
    return out


def _build_index() -> dict[str, str]:
    """Build the {symbol → include} index from disk."""
    index: dict[str, str] = dict(_SEED_SYMBOLS)
    gemmi_root = Path(GEMMI_INCLUDE) / "gemmi"
    gtest_root = Path(GTEST_INCLUDE)

    for header in gemmi_root.glob("*.hpp"):
        for sym, inc in _scan_header(header, gemmi_root, gtest_root):
            # Don't overwrite seed entries — they're authoritative for cases
            # where a symbol is forward-declared in multiple headers.
            index.setdefault(sym, inc)

    if gtest_root.exists():
        for header in gtest_root.rglob("*.h"):
            for sym, inc in _scan_header(header, gemmi_root, gtest_root):
                index.setdefault(sym, inc)

    return index


@lru_cache(maxsize=1)
def _load_index() -> dict[str, str]:
    """Lazy-load the index from disk; rebuild and persist on a miss."""
    if INDEX_PATH.exists():
        try:
            data = json.loads(INDEX_PATH.read_text())
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    index = _build_index()
    try:
        INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True))
    except OSError:
        pass
    return index


def include_for_symbol(symbol: str) -> str:
    """Return the canonical `#include` directive that defines `symbol`.

    Strips namespaces and trailing call syntax so callers can pass natural
    forms like 'gemmi::read_pdb_file(' or 'TEST(' and still get a hit.
    """
    sym = symbol.strip()
    if "(" in sym:
        sym = sym.split("(", 1)[0].strip()
    if "::" in sym:
        sym = sym.rsplit("::", 1)[-1]
    if sym.startswith(("&", "*")):
        sym = sym.lstrip("&*").strip()

    index = _load_index()
    inc = index.get(sym)
    if inc:
        return f"#include {inc}"

    # Suggest near matches before giving up — saves the agent a follow-up call.
    # Require at least 4 chars of overlap to avoid trivia like 'A' matching
    # everything single-letter.
    sym_l = sym.lower()
    if len(sym_l) >= 4:
        near = [s for s in index
                if (sym_l in s.lower() or s.lower() in sym_l)
                and abs(len(s) - len(sym_l)) <= max(4, len(sym_l) // 2)]
        if 0 < len(near) <= 8:
            lines = [f"No exact match for '{symbol}'. Similar symbols:"]
            lines.extend(f"  {s}: #include {index[s]}" for s in sorted(near))
            return "\n".join(lines)
    return (
        f"No header found for '{symbol}'. If this is a free function, "
        "try grep_codebase. If it's a member, lookup_type the parent class."
    )


def rebuild_index() -> int:
    """Force-rebuild the on-disk index. Returns the number of symbols."""
    INDEX_PATH.unlink(missing_ok=True)
    _load_index.cache_clear()
    return len(_load_index())
