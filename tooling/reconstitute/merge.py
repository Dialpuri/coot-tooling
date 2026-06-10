"""Assemble verified per-port lift fragments into the parallel `_gemmi` files.

Each passing port leaves `<port_dir>/reconstitute/function.hh` (a full
`class <class>_gemmi` carrying ONE method) plus an optional `function.cc`. This
module collects them for a class and emits:

    <out_dir>/<header_stem>_gemmi.hh        the single class, all method DECLs
    <out_dir>/<source_stem>_gemmi.cc        out-of-line DEFs, one file per
                                            original source file the methods
                                            came from

The merge is purely mechanical: every fragment already compiled, so the input is
guaranteed-valid C++ and a brace/paren-aware scanner extracts each method's
declaration and definition deterministically. A method that the agent inlined in
the class is split here into a class declaration + an out-of-line definition; a
method already split into `function.cc` is used as-is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import sqlite3

from ..gemmi.aggregate import _split_header, _dedup_ordered
from ..oracle.generate import OUT_ROOT, sanitize_name
from ..db import get_class_functions, group_entries_by_source_file
from .classmodel import ClassModel, class_model_for, STRUCTURE_MEMBER

_IDENT = re.compile(r"[A-Za-z_]\w*$")
_ACCESS_RE = re.compile(r"\b(public|private|protected)\s*:")


# ── brace / paren scanning ────────────────────────────────────────────────────

def _matching(text: str, i: int) -> int:
    """Given text[i] is one of '{([', return the index of its matching close.

    Skips over // and /* */ comments and "..." / '...' literals so braces inside
    them don't throw off the count. Raises ValueError if unmatched.
    """
    pairs = {"{": "}", "(": ")", "[": "]"}
    close = pairs[text[i]]
    opens = set(pairs)
    closes = set(pairs.values())
    depth = 0
    j = i
    n = len(text)
    while j < n:
        c = text[j]
        two = text[j:j + 2]
        if two == "//":
            j = text.find("\n", j)
            if j == -1:
                break
            continue
        if two == "/*":
            end = text.find("*/", j + 2)
            j = end + 2 if end != -1 else n
            continue
        if c in ('"', "'"):
            j = _skip_string(text, j)
            continue
        if c in opens:
            depth += 1
        elif c in closes:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    raise ValueError(f"unmatched {text[i]!r} at index {i}")


def _skip_string(text: str, i: int) -> int:
    """text[i] opens a string/char literal; return index just past its close."""
    quote = text[i]
    j = i + 1
    n = len(text)
    while j < n:
        if text[j] == "\\":
            j += 2
            continue
        if text[j] == quote:
            return j + 1
        j += 1
    return n


# ── per-port fragment parsing ─────────────────────────────────────────────────

@dataclass
class MethodPiece:
    qname: str
    decl: str            # in-class declaration, e.g. "unsigned int foo() const;"
    definition: str      # out-of-line def text (no namespace wrapper)


@dataclass
class PortFragment:
    qname: str
    includes: list[str]
    methods: list[MethodPiece]
    extra_members: list[str] = field(default_factory=list)
    helpers: list[str] = field(default_factory=list)   # namespace-level extras
    warnings: list[str] = field(default_factory=list)


def _find_class_body(body: str, class_name: str) -> tuple[int, int, int]:
    """Return (decl_start, body_open, body_close) for `class <class_name> { ... }`.

    decl_start is the index of the `class` keyword; body_open/close bracket the
    `{ ... }`. Raises ValueError if the class isn't found.
    """
    m = re.search(rf"\bclass\s+{re.escape(class_name)}\b", body)
    if not m:
        raise ValueError(f"class {class_name} not found")
    brace = body.find("{", m.end())
    if brace == -1:
        raise ValueError(f"no body for class {class_name}")
    return m.start(), brace, _matching(body, brace)


def _qualify(signature: str, class_name: str) -> str | None:
    """Insert `<class_name>::` before the method name in a signature.

    The method name is the identifier immediately before the first top-level
    '('. Returns None if it can't be located (operator overloads etc.).
    """
    paren = signature.find("(")
    if paren == -1:
        return None
    before = signature[:paren]
    m = _IDENT.search(before.rstrip())
    if not m:
        return None
    name_end = len(before.rstrip())
    name_start = name_end - len(m.group(0))
    return before[:name_start] + f"{class_name}::" + before[name_start:] + signature[paren:]


def _scan_class_members(
    class_body: str, class_name: str, cls: ClassModel,
) -> tuple[list[MethodPiece], list[str], list[str]]:
    """Walk a class body; return (methods, extra_member_decls, warnings).

    Constructors and the standard `structure` member are dropped (the merged
    class re-emits them from the class model). Data members other than those are
    reported as extras. Inline methods are split into decl + out-of-line def.
    """
    methods: list[MethodPiece] = []
    extras: list[str] = []
    warnings: list[str] = []
    i, n = 0, len(class_body)
    while i < n:
        # Skip whitespace / access specifiers / stray semicolons.
        while i < n and class_body[i] in " \t\r\n;":
            i += 1
        if i >= n:
            break
        acc = _ACCESS_RE.match(class_body, i)
        if acc:
            i = acc.end()
            continue
        # Read one member unit, tracking paren depth so ';'/'{' inside params or
        # default args don't end the unit early.
        start = i
        body_open = -1
        unit_end = -1
        while i < n:
            c = class_body[i]
            two = class_body[i:i + 2]
            if two == "//":
                i = class_body.find("\n", i)
                i = n if i == -1 else i
                continue
            if two == "/*":
                end = class_body.find("*/", i + 2)
                i = end + 2 if end != -1 else n
                continue
            if c in ('"', "'"):
                i = _skip_string(class_body, i)
                continue
            if c == "(" or c == "[":
                i = _matching(class_body, i)
                i += 1
                continue
            if c == "{":
                body_open = i
                close = _matching(class_body, i)
                unit_end = close + 1
                i = unit_end
                break
            if c == ";":
                unit_end = i + 1
                i = unit_end
                break
            i += 1
        unit = class_body[start:unit_end].strip()
        if not unit:
            continue
        self_paren = unit.find("(")
        if self_paren == -1:
            # Data member. Keep extras (anything but the standard structure one).
            norm = re.sub(r"\s+", " ", unit)
            if not re.match(rf"gemmi::Structure\s+{STRUCTURE_MEMBER}\s*;", norm):
                extras.append(unit)
            continue
        # Has a parameter list → ctor or method.
        name_before = unit[:self_paren].rstrip()
        nm = _IDENT.search(name_before)
        name = nm.group(0) if nm else ""
        if name == class_name or name == f"~{class_name}" or name.endswith(class_name):
            continue  # constructor / destructor — re-emitted from the model
        if body_open != -1:
            sig = class_body[start:body_open].strip()
            body = class_body[body_open:unit_end].rstrip().rstrip(";").rstrip()
            qualified = _qualify(sig, class_name)
            if qualified is None:
                warnings.append(f"could not qualify inline method: {sig[:80]}")
                continue
            decl = sig + ";"
            definition = f"{qualified} {body}"
        else:
            # Declaration only — the definition lives in function.cc.
            decl = unit if unit.endswith(";") else unit + ";"
            definition = ""
        methods.append(MethodPiece(qname="", decl=decl, definition=definition))
    return methods, extras, warnings


def parse_port_fragment(port_dir: Path, cls: ClassModel, qname: str) -> PortFragment:
    """Parse one port's reconstitute/{function.hh,function.cc} into a fragment."""
    recon = port_dir / "reconstitute"
    hh_text = (recon / "function.hh").read_text()
    includes, hh_body = _split_header(hh_text)

    decl_start, body_open, body_close = _find_class_body(hh_body, cls.class_name)
    class_body = hh_body[body_open + 1:body_close]
    # Namespace-level code outside the class (helpers, free functions). Skip the
    # class declaration's own trailing `;` so it isn't captured as a helper.
    after = hh_body[body_close + 1:].lstrip()
    if after.startswith(";"):
        after = after[1:]
    outside = hh_body[:decl_start] + after
    helpers: list[str] = []
    helper_text = _namespace_inner(outside, cls.namespace)
    if _is_real_helper(helper_text):
        helpers.append(helper_text)

    methods, extras, warnings = _scan_class_members(class_body, cls.class_name, cls)
    for mp in methods:
        mp.qname = qname

    # Out-of-line defs supplied by function.cc replace empty (decl-only) defs.
    cc_path = recon / "function.cc"
    if cc_path.exists():
        cc_inc, cc_body = _split_header(cc_path.read_text())
        includes.extend(cc_inc)
        cc_defs = _namespace_inner(cc_body, cls.namespace).strip()
        if cc_defs:
            # Attach the whole .cc body to the first method that lacks a def.
            placed = False
            for mp in methods:
                if not mp.definition:
                    mp.definition = cc_defs
                    placed = True
                    break
            if not placed and _is_real_helper(cc_defs):
                helpers.append(cc_defs)

    return PortFragment(
        qname=qname, includes=includes, methods=methods,
        extra_members=extras, helpers=helpers, warnings=warnings,
    )


def _is_real_helper(text: str) -> bool:
    """True if `text` carries actual code, not just stray punctuation/whitespace."""
    return bool(re.sub(r"[\s;]", "", text or ""))


def _namespace_inner(text: str, namespace: str) -> str:
    """Return the inside of `namespace <ns> { ... }`, or the text unchanged."""
    m = re.search(rf"\bnamespace\s+{re.escape(namespace)}\b", text)
    if not m:
        return text.strip()
    brace = text.find("{", m.end())
    if brace == -1:
        return text.strip()
    close = _matching(text, brace)
    return text[brace + 1:close].strip()


# ── assembly ──────────────────────────────────────────────────────────────────

_LICENSE = "// Generated by tooling.reconstitute — gemmi port of {orig}.\n"


def render_header(cls: ClassModel, header_stem: str,
                  fragments: list[PortFragment]) -> str:
    """Build the single `<header_stem>_gemmi.hh` declaring the whole class."""
    guard = cls.header_guard(header_stem)
    includes: list[str] = []
    method_decls: list[str] = []
    helper_decls: list[str] = []
    for fr in fragments:
        includes.extend(fr.includes)
        for em in fr.extra_members:
            cls.extra_members.setdefault(_member_name(em), em)
        for mp in fr.methods:
            method_decls.append(mp.decl)

    lines = [f"#ifndef {guard}", f"#define {guard}", ""]
    lines.extend(_dedup_ordered(includes))
    lines.append("")
    lines.append(f"namespace {cls.namespace} {{")
    lines.append("")
    lines.append(f"class {cls.class_name} {{")
    lines.append("public:")
    for m in cls.member_decls():
        lines.append(f"   {m}")
    lines.append("")
    for c in cls.constructor_decls():
        lines.append(f"   {c}")
    lines.append("")
    for d in _dedup_ordered(method_decls):
        lines.append(f"   {d}")
    lines.append("};")
    lines.append("")
    lines.append(f"}}  // namespace {cls.namespace}")
    lines.append("")
    lines.append(f"#endif  // {guard}")
    lines.append("")
    return "\n".join(lines)


def render_source(cls: ClassModel, source_stem: str, header_name: str,
                  fragments: list[PortFragment]) -> str | None:
    """Build one `<source_stem>_gemmi.cc` with the out-of-line definitions.

    Returns None if none of the fragments contributed a definition.
    """
    includes: list[str] = []
    defs: list[str] = []
    helpers: list[str] = []
    for fr in fragments:
        includes.extend(fr.includes)
        helpers.extend(fr.helpers)
        for mp in fr.methods:
            if mp.definition:
                defs.append(f"// --- {mp.qname} ---\n{mp.definition}")
    if not defs and not helpers:
        return None

    # Includes already pulled into the header are not repeated here.
    lines = [f'#include "{header_name}"', ""]
    for inc in _dedup_ordered(includes):
        lines.append(inc)
    lines.append("")
    lines.append(f"namespace {cls.namespace} {{")
    lines.append("")
    for h in _dedup_ordered(helpers):
        lines.append(h)
        lines.append("")
    lines.append("\n\n".join(defs))
    lines.append("")
    lines.append(f"}}  // namespace {cls.namespace}")
    lines.append("")
    return "\n".join(lines)


def _member_name(decl: str) -> str:
    """Best-effort member identifier from a declaration line (for dedup keys)."""
    m = re.search(r"(\w+)\s*(=|;|\[)", decl)
    return m.group(1) if m else decl.strip()


# ── orchestration ─────────────────────────────────────────────────────────────

def _header_stem_for(class_qname: str, source_files: list[str]) -> str:
    """Pick the single header stem for the class.

    The whole class shares one header (the user's "one header per class"). We
    name it after the primary source file — the one defining the most methods —
    which for molecule_t is `coot-molecule`.
    """
    if not source_files:
        return class_qname.rsplit("::", 1)[-1]
    return Path(source_files[0]).stem


@dataclass
class MergeResult:
    header_path: Path
    source_paths: list[Path]
    warnings: list[str]
    extra_members: dict[str, str]


def merge_class(
    conn: sqlite3.Connection,
    class_qname: str,
    out_dir: Path | None = None,
) -> MergeResult:
    """Merge every passing reconstituted port of a class into `_gemmi` files.

    Output (staging) layout under `out_dir` (default generated-tests/_reconstituted/):
        <header_stem>_gemmi.hh         the class
        <source_stem>_gemmi.cc         one per original source file
    The coot-tree placement + build registration is commit.py's job.
    """
    cls = class_model_for(class_qname)
    out_dir = out_dir or (OUT_ROOT / "_reconstituted")
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = get_class_functions(conn, class_qname, mmdb_only=True)
    # Keep only ports with a verified lift.
    passing = [
        (qn, sh) for qn, sh in entries
        if (OUT_ROOT / sanitize_name(qn, sh) / "reconstitute" / "reconstitute_ok").exists()
    ]
    buckets = group_entries_by_source_file(conn, passing)
    # Largest bucket first → its stem names the shared header.
    ordered_files = sorted(buckets, key=lambda f: (-len(buckets[f]), f))
    header_stem = _header_stem_for(class_qname, ordered_files)
    header_name = f"{header_stem}_gemmi.hh"

    # Parse every fragment, grouped by source file.
    frags_by_file: dict[str, list[PortFragment]] = {}
    all_frags: list[PortFragment] = []
    warnings: list[str] = []
    for src in ordered_files:
        for qn, sh in buckets[src]:
            port_dir = OUT_ROOT / sanitize_name(qn, sh)
            fr = parse_port_fragment(port_dir, cls, qn)
            frags_by_file.setdefault(src, []).append(fr)
            all_frags.append(fr)
            for w in fr.warnings:
                warnings.append(f"{qn}: {w}")

    # Header sees every fragment (all declarations + unioned members).
    header_text = render_header(cls, header_stem, all_frags)
    header_path = out_dir / header_name
    header_path.write_text(header_text)

    # One .cc per source file.
    source_paths: list[Path] = []
    for src in ordered_files:
        src_stem = Path(src).stem
        cc_text = render_source(cls, src_stem, header_name, frags_by_file[src])
        if cc_text is None:
            continue
        cc_path = out_dir / f"{src_stem}_gemmi.cc"
        cc_path.write_text(cc_text)
        source_paths.append(cc_path)

    return MergeResult(
        header_path=header_path,
        source_paths=source_paths,
        warnings=warnings,
        extra_members=dict(cls.extra_members),
    )
