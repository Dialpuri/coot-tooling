"""Pre-compile static linter for gemmi anti-patterns.

Each pattern in `_PATTERNS` matches a recurring mistake the agent makes when
porting MMDB → gemmi. Run before `compile_gemmi` so the agent gets instant
feedback on cheap-to-detect errors and doesn't burn a compile attempt on them.

Patterns are derived from real failure data in generated-tests/*/gemmi/compile.log
— each one was responsible for at least 3 verify-stage compile failures across
the 180-function corpus.
"""
from __future__ import annotations

import re

# (regex, fix-message). Keep the regex tight: a false positive here costs the
# agent a compile attempt to disprove, so prefer specificity over recall.
_PATTERNS: list[tuple[str, str]] = [
    (r"\bgemmi::Real3\b",
     "gemmi::Real3 doesn't exist — use gemmi::Vec3 (raw 3-vector) "
     "or gemmi::Position (Vec3 in Cartesian Å)."),
    (r"\bgemmi::vec3\b",
     "lowercase gemmi::vec3 doesn't exist — use gemmi::Vec3 (capital V)."),
    (r"\bgemmi::Cell\b(?!ular)",
     "gemmi::Cell doesn't exist — use gemmi::UnitCell."),
    (r"\.alt_loc\b",
     "gemmi::Atom field is `altloc` (no underscore), not `alt_loc`."),
    # Residue::add_atom doesn't exist; mmdb's AddAtom maps to push_back.
    (r"\.add_atom\s*\(",
     "gemmi::Residue has no add_atom() method — use "
     "`residue.atoms.push_back(atom)` or `emplace_back`."),
    (r"\bst\s*\.\s*setup_entities\s*\(",
     "setup_entities is a free function, not a method — call "
     "`gemmi::setup_entities(st)` and #include <gemmi/polyheur.hpp>."),
    (r"\bgemmi::Element::[A-Z][a-z]?\b",
     "gemmi::Element has no enum constants — construct from a symbol string: "
     "`gemmi::Element(\"C\")` (or `gemmi::El::C` if you really want the enum, "
     "but Element(\"C\") is canonical)."),
    # Structure field name slips
    (r"\bst\s*\.\s*space_group\b",
     "gemmi::Structure field is `spacegroup_hm` (Hermann–Mauguin string), "
     "not `space_group`."),
    # connections vs links — only flag when used on a Structure-like name.
    (r"\b(?:st|structure)\s*\.\s*links\b",
     "gemmi::Structure field is `connections` (std::vector<Connection>), "
     "not `links`."),
    # NOTE: the following are real agent failures (seen in the compile-fail
    # corpus) that we deliberately DON'T regex, because catching them needs
    # variable-type tracking and a blind pattern would flag valid code. They
    # belong in the prose anti-pattern catalog (system prompt) instead:
    #   • ResidueId.num / ResidueId.chain — the seq number is `res.seqid.num`
    #     (an OptionalInt, `.num.value`); ResidueId has no `chain`.
    #   • Fractional.u/v/w — Fractional/Position are Vec3, fields are x/y/z.
    #   • cif::Loop.data / cif::Loop.find_value — Loop stores `tags` +
    #     flat `values`; read with `loop.val(row, col)` / `loop.find_tag(tag)`.
    #     `find_value` is real but lives on cif::Block, not cif::Loop — so a
    #     blanket `.find_value` rule would flag the valid Block call.
    #   • Atom.altLoc — gemmi's field is `altloc` (lowercase, a single char),
    #     but coot's own `minimol::atom::altLoc` is a legit member, so a bare
    #     `.altLoc` rule false-positives on minimol ports.
    # ApplyTransform / mat44 are MMDB names; flag if they leak into gemmi code.
    (r"\bgemmi::mat44\b",
     "gemmi has no mat44 — use gemmi::Transform (Mat33 + Vec3) "
     "from <gemmi/math.hpp>."),
    # Residue with parent pointer — flag direct .chain access on residue/Residue
    (r"(?:residue|res)\s*[.\->]+\s*chain\b(?!_id|\.name)",
     "gemmi::Residue has no parent pointer — there's no `residue.chain`. "
     "Pass a `gemmi::CRA{Chain*, Residue*, Atom*}` (the idiomatic carrier for "
     "parent context) or pair Chain* with Residue* during iteration via "
     "`for (auto& chain : model.chains) for (auto& res : chain.residues)`."),
    # `subchain` confusion: agents reach for r->subchain to recover the chain
    # name and get gemmi's auto-assigned polymer label (e.g. "Axp") instead.
    # Flag whenever subchain participates in a comparison or equality test.
    (r"(?:->|\.)subchain\s*(?:[<>!=]=?|==)",
     "Suspected chain-name confusion: `Residue::subchain` is being compared "
     "as if it were the chain ID. `subchain` is gemmi's auto-assigned "
     "polymer/entity label (e.g. \"Axp\" for chain \"A\"), NOT the "
     "user-visible chain name. Use the parent `Chain::name` — pass a "
     "`gemmi::CRA` (Chain*, Residue*, Atom* — all pointers) or pair Chain* "
     "with Residue* during iteration."),
    (r"EXPECT_(?:EQ|NE|STREQ|STRNE)\s*\([^,;)]*?(?:->|\.)subchain\b",
     "EXPECT_* on `Residue::subchain` is almost certainly the wrong field — "
     "subchain is gemmi's polymer/entity label (e.g. \"Axp\"), not the chain "
     "ID. Compare against the parent `Chain::name` instead (carry it via a "
     "`gemmi::CRA` or alongside the Residue*)."),
    # The old GCC ABI define belongs in oracle/test stages (MMDB linkage) but
    # must never appear in gemmi code — gemmi is compiled with the new ABI and
    # the mismatch causes silent linker failures on any std::string parameter.
    (r"#\s*define\s+_GLIBCXX_USE_CXX11_ABI\s+0",
     "#define _GLIBCXX_USE_CXX11_ABI 0 must NOT appear in gemmi-stage files. "
     "gemmi and libgemmi_cpp are compiled with the new (default) C++11 ABI; "
     "forcing ABI 0 causes link-time 'undefined reference' errors on any "
     "function whose signature contains std::string. Remove this line entirely."),
    # ── batch-mined 2026-05-21: header-verified, 0 false positives across the
    #    329 passing ports in generated-tests/. Each fired in the compile-fail
    #    corpus; replacements checked against ~/gemmi/include/gemmi.
    # gemmi::Atom has no parent back-pointer (unlike mmdb::Atom::residue).
    (r"\batom\s*(?:->|\.)\s*(?:residue|parent_residue|parent)\b",
     "gemmi::Atom has no parent pointer — there is no `atom.residue` / "
     "`atom.parent` / `atom.parent_residue`. Carry parent context with a "
     "`gemmi::CRA{Chain*, Residue*, Atom*}`, or pair Chain*/Residue*/Atom* "
     "during nested iteration "
     "(`for (auto& ch : model.chains) for (auto& res : ch.residues) "
     "for (auto& at : res.atoms)`)."),
    # Connection endpoints are partner1/partner2, not a `partners` container.
    (r"(?:->|\.)partners\b",
     "gemmi::Connection has no `partners` member — the two endpoints are "
     "`partner1` and `partner2` (each a `gemmi::AtomAddress`)."),
    # connections live on Structure, not Model.
    (r"(?:model|mdl)\s*(?:->|\.)\s*connections\b",
     "gemmi::Model has no `connections` — LINK/SSBOND connections live on "
     "`gemmi::Structure::connections` (std::vector<Connection>), not on Model."),
    # Element placeholder test: there's no is_dummy().
    (r"(?:element|elem|el)\s*(?:->|\.)\s*is_dummy\s*\(",
     "gemmi::Element has no is_dummy() — the unknown/placeholder element is "
     "`gemmi::El::X`; test with `el == gemmi::El::X` (Element compares against "
     "El). #include <gemmi/elem.hpp>."),
    # I/O free-function name slips.
    (r"\bgemmi::read_cif_file\b",
     "gemmi::read_cif_file doesn't exist — read a CIF document with "
     "`gemmi::cif::read_file(path)` (#include <gemmi/cif.hpp>), or read a "
     "model with `gemmi::read_structure` (#include <gemmi/mmread.hpp>)."),
    (r"\bgemmi::write_pdb_file\b",
     "gemmi::write_pdb_file doesn't exist — use "
     "`gemmi::write_pdb(structure, std::ostream&)` (#include <gemmi/to_pdb.hpp>)."),
    (r"\bgemmi::to_mmcif\b",
     "gemmi has no `gemmi::to_mmcif` — build an mmCIF document with "
     "`gemmi::make_mmcif_document(structure)` (#include <gemmi/to_mmcif.hpp>), "
     "then write it (`doc.write_file(path)`)."),
    # Geometry: dihedral is a free function; length/distance are methods.
    (r"\bgemmi::dihedral\s*\(",
     "gemmi::dihedral doesn't exist — use "
     "`gemmi::calculate_dihedral(p0, p1, p2, p3)` (four gemmi::Position, "
     "#include <gemmi/calculate.hpp>), or "
     "`gemmi::calculate_dihedral_from_atoms(a, b, c, d)`."),
    (r"\bgemmi::length\s*\(",
     "gemmi::length is not a free function — `length()` is a method: "
     "`v.length()` on a gemmi::Vec3/gemmi::Position (#include <gemmi/math.hpp>)."),
    (r"\bgemmi::distance\s*\(",
     "gemmi::distance doesn't exist — distance is a method: `p1.dist(p2)` on "
     "a gemmi::Position/gemmi::Vec3. For periodic distances use "
     "`UnitCell::find_nearest_image`."),
]


# Symbol → header pairs. If the symbol is used but the header is not included,
# the compile will fail with "is not a member of gemmi". Catch instantly.
_SYMBOL_HEADERS: list[tuple[str, str]] = [
    (r"\bgemmi::read_pdb_file\b",      "<gemmi/pdb.hpp>"),
    (r"\bgemmi::read_structure\b",     "<gemmi/mmread.hpp>"),
    (r"\bgemmi::read_ccp4_map\b",      "<gemmi/ccp4.hpp>"),
    (r"\bgemmi::read_mtz_file\b",      "<gemmi/mtz.hpp>"),
    (r"\bgemmi::NeighborSearch\b",     "<gemmi/neighbor.hpp>"),
    (r"\bgemmi::ContactSearch\b",      "<gemmi/contact.hpp>"),
    (r"\bgemmi::setup_entities\b",     "<gemmi/polyheur.hpp>"),
    (r"\bgemmi::remove_waters\b",      "<gemmi/polyheur.hpp>"),
    (r"\bgemmi::remove_alternative_conformations\b", "<gemmi/modify.hpp>"),
    (r"\bgemmi::remove_empty_children\b",            "<gemmi/modify.hpp>"),
    (r"\bgemmi::transform_pos_and_adp\b",            "<gemmi/modify.hpp>"),
    (r"\bgemmi::write_pdb\b",          "<gemmi/to_pdb.hpp>"),
    (r"\bgemmi::write_cif\b",          "<gemmi/to_cif.hpp>"),
    (r"\bgemmi::find_tabulated_residue\b",           "<gemmi/resinfo.hpp>"),
    (r"\bgemmi::calculate_center_of_mass\b",         "<gemmi/calculate.hpp>"),
    (r"\bgemmi::Element\b",            "<gemmi/elem.hpp>"),
    (r"\bgemmi::Mtz\b",                "<gemmi/mtz.hpp>"),
    (r"\bgemmi::Grid\b",               "<gemmi/grid.hpp>"),
    (r"\bgemmi::DsspCalculator\b",     "<gemmi/dssp.hpp>"),
    (r"\bgemmi::make_assembly\b",      "<gemmi/assembly.hpp>"),
    (r"\bgemmi::SmallStructure\b",     "<gemmi/small.hpp>"),
    (r"\bgemmi::make_mmcif_document\b", "<gemmi/to_mmcif.hpp>"),
    (r"\bgemmi::calculate_dihedral\b", "<gemmi/calculate.hpp>"),
    (r"\bgemmi::calculate_angle\b",    "<gemmi/calculate.hpp>"),
    (r"\bTEST\s*\(",                   "<gtest/gtest.h>"),
    (r"\bEXPECT_(?:EQ|NE|TRUE|FALSE|FLOAT_EQ|DOUBLE_EQ|NEAR|LT|LE|GT|GE)\b",
                                       "<gtest/gtest.h>"),
]


def _missing_header_findings(code: str) -> list[str]:
    """For every gemmi symbol used, verify its required header is #included.

    Returns one finding per (symbol-use, missing-header) pair, deduped by header
    so repeated uses don't spam the report.
    """
    out: list[str] = []
    seen_headers: set[str] = set()
    # Collect every #include in the file once.
    includes_in_file = set(re.findall(r'#\s*include\s+([<"][^>"]+[>"])', code))
    for sym_pat, header in _SYMBOL_HEADERS:
        if header in seen_headers:
            continue
        # Is this header already there (in either bracket or quote form)?
        if any(header.strip("<>") in inc for inc in includes_in_file):
            continue
        m = re.search(sym_pat, code)
        if m:
            line_no = code.count("\n", 0, m.start()) + 1
            out.append(
                f"line {line_no}: missing #include {header} "
                f"(needed for the symbol matched here)"
            )
            seen_headers.add(header)
    return out


# ── redefinition of real coot types ──────────────────────────────────────────
#
# The single worst failure mode: a port stubs out a real coot type
# (`struct simple_mesh_t { ... }`) instead of #including its header, so the body
# can fabricate a result that games the frozen assertions. It compiles in
# isolation but is wrong, and the stub leaks into the reconstituted class. This
# rule forbids redefining any of coot's OWN record types (scoped to `coot::` so
# std/gemmi/clipper short-name collisions can't false-positive) and points the
# agent at the real header instead.

# A `struct`/`class Name {` definition (optional `final` / base-clause). A bare
# forward decl (`class Foo;`) has no `{` and is not matched.
_TYPE_DEF_RE = re.compile(
    r"\b(?:struct|class)\s+([A-Za-z_]\w*)\s*(?:final\s*)?(?::[^{;]*)?\{")

# Generic single-word names that legitimately recur as local helpers; too risky
# to hard-block even though a coot:: type shares the name.
_GENERIC_TYPE_NAMES = frozenset({
    "atom", "bond", "residue", "chain", "link", "node", "edge", "cell",
    "point", "line", "range", "result", "state", "data", "info", "item",
})

# Lazily-built {short_name: header_path}; one DB, so a module-level cache is fine.
_COOT_TYPE_INDEX: dict[str, str] | None = None


def coot_record_type_index(conn) -> dict[str, str]:
    """Map coot's own record type short-names -> the header that declares them.

    Used by the redefinition rule to (a) recognise a stubbed coot type and
    (b) suggest the include that replaces the stub. Cached across calls.
    """
    global _COOT_TYPE_INDEX
    if _COOT_TYPE_INDEX is not None:
        return _COOT_TYPE_INDEX
    rows = conn.execute("""
        SELECT t.qualified_name, fi.path
        FROM types t JOIN files fi ON fi.id = t.file_id
        WHERE t.kind IN ('CLASS_DECL', 'STRUCT_DECL', 'CLASS_TEMPLATE',
                         'class', 'struct')
          AND t.qualified_name LIKE 'coot::%'
    """).fetchall()
    idx: dict[str, str] = {}
    for qn, path in rows:
        short = qn.rsplit("::", 1)[-1]
        if len(short) >= 4 and re.match(r"^[A-Za-z_]\w*$", short):
            idx.setdefault(short, path)
    _COOT_TYPE_INDEX = idx
    return idx


def _tree_relative(path: str) -> str:
    """Best-effort coot-tree-relative include path for a header on disk."""
    from ..db import PROJECT_ROOT
    if path.startswith(PROJECT_ROOT + "/"):
        return path[len(PROJECT_ROOT) + 1:]
    return path.rsplit("/", 1)[-1]


def redefined_coot_type_findings(code: str, type_index: dict[str, str]) -> list[str]:
    """Flag stub redefinitions of real coot types that lack the real include.

    Only fires when the header is NOT already included — if the agent both
    stubs and includes the real type, the compiler's redefinition error is the
    clearer signal, so we leave that to compile_gemmi.
    """
    out: list[str] = []
    includes = code  # cheap substring test below against the whole file
    for m in _TYPE_DEF_RE.finditer(code):
        name = m.group(1)
        if name.lower() in _GENERIC_TYPE_NAMES:
            continue
        header = type_index.get(name)
        if not header:
            continue
        rel = _tree_relative(header)
        if rel.rsplit("/", 1)[-1] in includes:        # real header already in
            continue
        line_no = code.count("\n", 0, m.start()) + 1
        out.append(
            f"line {line_no}: `{m.group(0).rstrip('{').strip()}` redefines the "
            f"real coot type `coot::{name}` — DELETE this stub and "
            f'`#include "{rel}"` instead. A local copy compiles here but '
            f"fabricates the type, lets the body fake the result, and collides "
            f"the moment the port is reconstituted."
        )
    return out


def gemmi_lint(code: str, coot_types: dict[str, str] | None = None) -> list[str]:
    """Return a list of human-readable findings, empty if clean.

    Each finding is `"line N: <fix-message>"` so the agent can locate the
    offending line directly. When `coot_types` (a `coot_record_type_index`) is
    supplied, the real-coot-type redefinition rule is enabled too.
    """
    findings: list[str] = []
    for pat, fix in _PATTERNS:
        try:
            rx = re.compile(pat)
        except re.error:
            continue
        for m in rx.finditer(code):
            line_no = code.count("\n", 0, m.start()) + 1
            findings.append(f"line {line_no}: {fix}")
    findings.extend(_missing_header_findings(code))
    if coot_types:
        findings.extend(redefined_coot_type_findings(code, coot_types))
    # Dedup while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for f in findings:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


def lint_report(code: str) -> str:
    """Render a `gemmi_lint` result for inclusion in a tool response."""
    findings = gemmi_lint(code)
    if not findings:
        return "OK — no gemmi anti-patterns detected."
    return ("Lint findings (fix these BEFORE compile_gemmi — "
            "this does NOT count against your compile budget):\n"
            + "\n".join(f"  - {f}" for f in findings))


def has_lint_findings(code: str) -> bool:
    return bool(gemmi_lint(code))
