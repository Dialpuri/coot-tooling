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
from ..db import (
    get_class_functions, group_entries_by_source_file, PROJECT_ROOT,
    get_class_methods_with_access,
)
from .classmodel import ClassModel, class_model_for, STRUCTURE_MEMBER

_IDENT = re.compile(r"[A-Za-z_]\w*$")
_ACCESS_RE = re.compile(r"\b(public|private|protected)\s*:")
_INCLUDE_PATH_RE = re.compile(r'#\s*include\s*[<"]([^>"]+)[>"]')


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

    decl_start is the index of the `class`/`struct` keyword; body_open/close
    bracket the `{ ... }`. The agent sometimes emits `struct` instead of `class`
    (equivalent in C++ bar default access), so accept either. Raises ValueError
    if the class isn't found.
    """
    m = re.search(rf"\b(?:class|struct)\s+{re.escape(class_name)}\b", body)
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


# ── lint / cleanup ────────────────────────────────────────────────────────────

def _clean_includes(includes: list[str], method_port_prefix: str = "") -> list[str]:
    """Drop sibling-method port includes, keep free-fn deps, dedupe the rest.

    Per-port fragments carry phase-1 port includes
    (`generated-tests/.../gemmi/function.hh`, coot-tree `.../gemmi/.../function.hh`
    ports, and a bare `#include "function.hh"`). Two kinds:

    * a SIBLING method of this class (port dir starts with `method_port_prefix`,
      e.g. `coot__molecule_t__cid_to_atom`) — dropped, because that method now
      lives in the merged class and is called via `this->`;
    * a FREE-FUNCTION dependency in another namespace (`coot__util__...`,
      `make_asc`, ...) — KEPT, because the merged methods still call it and its
      definition lives only in that port header.

    Absolute coot-tree includes are rewritten to tree-relative so they collapse
    with relative twins; trailing line comments are stripped so
    `<utility>  // for pair` dedupes against `<utility>`.
    """
    out: list[str] = []
    for inc in includes:
        m = _INCLUDE_PATH_RE.search(inc)
        if not m:
            continue
        path = m.group(1)
        if path == "function.hh":             # dangling relative self-include
            continue
        is_port = ("generated-tests/" in path
                   or ("/gemmi/" in path and path.endswith("function.hh")))
        if is_port:
            # Sibling-method ports are now in-class; free-fn dep ports stay.
            if method_port_prefix and method_port_prefix in path:
                continue
            if not method_port_prefix:        # legacy: no prefix → drop all ports
                continue
            out.append(f'#include "{path}"')
            continue
        # Absolute coot-tree include → tree-relative, so it dedupes.
        if path.startswith(PROJECT_ROOT + "/"):
            path = path[len(PROJECT_ROOT) + 1:]
        bracket = "<" in inc.split("include", 1)[1].lstrip()[:1]
        out.append(f"#include <{path}>" if bracket else f'#include "{path}"')
    return _dedup_ordered(out)


def _strip_comments(s: str) -> str:
    """Remove // and /* */ comments — decls carry doc comments that contain
    stray '(' and identifiers which otherwise derail signature parsing."""
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//[^\n]*", "", s)
    return s


def _param_types(decl: str) -> str | None:
    """Normalized parameter-type list (names/defaults stripped) for a signature.

    Used as part of the overload key: two decls with the same name, the same
    param types and the same const-ness are the same method, regardless of the
    parameter names the agent happened to choose. Returns None if the parameter
    list can't be located.
    """
    open_paren = decl.find("(")
    if open_paren == -1:
        return None
    try:
        close = _matching(decl, open_paren)
    except ValueError:
        return None       # malformed decl — leave it untouched, don't dedupe
    inner = decl[open_paren + 1:close]
    # Split on top-level commas only (templates carry their own commas).
    parts, depth, start = [], 0, 0
    for i, c in enumerate(inner):
        if c in "<([":
            depth += 1
        elif c in ">)]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(inner[start:i])
            start = i + 1
    parts.append(inner[start:])

    norm = []
    for p in parts:
        p = re.sub(r"/\*.*?\*/", "", p)          # strip /* */ comments
        p = p.split("=", 1)[0].strip()           # strip default value
        if not p:
            continue
        # Drop a trailing parameter name: a final identifier whose preceding
        # type-part ends in something that can't be a namespace separator.
        m = re.search(r"([A-Za-z_]\w*)\s*$", p)
        if m:
            head = p[:m.start()].rstrip()
            if head and not head.endswith(("::", "<")):
                p = head
        p = re.sub(r"\s*::\s*", "::", p)
        p = re.sub(r"\s*([*&,])\s*", r"\1", p)
        p = re.sub(r"\s+", " ", p).strip()
        norm.append(p)
    return ",".join(norm)


def _overload_key(decl: str) -> tuple[str, str, bool] | None:
    """(name, normalized-params, is_const) — same key ⇒ same overload slot."""
    clean = _strip_comments(decl)
    paren = clean.find("(")
    if paren == -1:
        return None
    nm = _IDENT.search(clean[:paren].rstrip())
    if not nm:
        return None
    params = _param_types(clean)
    if params is None:
        return None
    is_const = bool(re.search(r"\)\s*const\b", clean))
    return nm.group(0), params, is_const


def _decl_method_name(decl: str) -> str | None:
    """The method identifier of a declaration (comment-stripped), or None."""
    clean = _strip_comments(decl)
    paren = clean.find("(")
    if paren == -1:
        return None
    nm = _IDENT.search(clean[:paren].rstrip())
    return nm.group(0) if nm else None


def _access_of(decl: str, access_by_name: dict[str, str | None]) -> str:
    """Resolve the access level for a method declaration.

    A method matching a real DB method keeps that method's access (an
    undetermined `None` from libclang is treated as public — it's a real API
    method). A method NOT in the class (a helper the lift invented) is made
    private: invented internals should not widen the class's public surface.
    """
    name = _decl_method_name(decl)
    if name is None:
        return "public"               # not a method (helper/typedef) — leave public
    if name in access_by_name:
        return access_by_name[name] or "public"
    return "private"


def _return_prefix(decl: str, name: str) -> str:
    """Normalized return type + specifiers of a decl, for conflict comparison.

    Strips leading doc comments and collapses pointer/reference/whitespace so
    `gemmi::Atom *foo` and `gemmi::Atom* foo` compare equal (pure duplicate),
    while `int undo` vs `gemmi::Structure undo` stay distinct (real conflict).
    """
    clean = _strip_comments(decl)
    head = clean[:clean.find("(")]
    head = head.rsplit(name, 1)[0]
    head = re.sub(r"\s*([*&])\s*", r"\1", head)
    return re.sub(r"\s+", " ", head).strip()


def _dedup_methods(decls: list[str]) -> tuple[list[str], list[str]]:
    """Dedupe method declarations by overload signature.

    Exact-string dedup leaves trivially-conflicting redeclarations (same name +
    params + const-ness but a different return type) and pure duplicates that
    differ only in parameter names. Here the first decl in each overload slot
    wins; an identical-after-normalization repeat is dropped silently, and a
    repeat with a different return type/specifier is dropped with a warning so
    the divergence is reviewed rather than emitted as uncompilable code.
    """
    kept: list[str] = []
    warnings: list[str] = []
    first: dict[tuple[str, str, bool], str] = {}
    for d in decls:
        key = _overload_key(d)
        if key is None:                  # not a method decl — keep as-is
            kept.append(d)
            continue
        if key not in first:
            first[key] = d
            kept.append(d)
            continue
        # Same overload slot already taken. Conflict only if the return
        # type / specifiers differ (param-name-only diffs are pure dups).
        prev = first[key]
        prev_ret = _return_prefix(prev, key[0])
        cur_ret = _return_prefix(d, key[0])
        if prev_ret != cur_ret:
            warnings.append(
                f"conflicting decl for {key[0]}({key[1]}): kept "
                f"'{prev_ret} {key[0]}', dropped '{cur_ret} {key[0]}'"
            )
    return kept, warnings


# ── assembly ──────────────────────────────────────────────────────────────────

_LICENSE = "// Generated by tooling.reconstitute — gemmi port of {orig}.\n"


def render_header(cls: ClassModel, header_stem: str,
                  fragments: list[PortFragment],
                  warnings: list[str] | None = None,
                  access_by_name: dict[str, str | None] | None = None,
                  method_port_prefix: str = "") -> str:
    """Build the single `<header_stem>_gemmi.hh` declaring the whole class.

    Methods are grouped under `public:`/`protected:`/`private:` mirroring the
    original MMDB class's access (from `access_by_name`); members and the
    construction contract stay public. Access is applied HERE, not in the lift —
    a per-port lift's own test calls the method directly, so the lifted fragment
    must keep it public to self-verify; the merge is what restores fidelity.
    """
    guard = cls.header_guard(header_stem)
    access_by_name = access_by_name or {}
    includes: list[str] = []
    method_decls: list[str] = []
    helper_decls: list[str] = []
    for fr in fragments:
        includes.extend(fr.includes)
        for em in fr.extra_members:
            cls.extra_members.setdefault(_member_name(em), em)
        for mp in fr.methods:
            method_decls.append(mp.decl)

    decls, dedup_warnings = _dedup_methods(_dedup_ordered(method_decls))
    if warnings is not None:
        warnings.extend(dedup_warnings)

    # Group by access; public first so the class reads API-first.
    grouped: dict[str, list[str]] = {"public": [], "protected": [], "private": []}
    for d in decls:
        grouped[_access_of(d, access_by_name)].append(d)

    lines = [f"#ifndef {guard}", f"#define {guard}", ""]
    lines.extend(_clean_includes(includes, method_port_prefix))
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
    # Public methods continue the existing public section (members + ctors);
    # protected/private get their own labels.
    if grouped["public"]:
        lines.append("")
        for d in grouped["public"]:
            lines.append(f"   {d}")
    for access in ("protected", "private"):
        if not grouped[access]:
            continue
        lines.append("")
        lines.append(f"{access}:")
        for d in grouped[access]:
            lines.append(f"   {d}")
    lines.append("};")
    lines.append("")
    lines.append(f"}}  // namespace {cls.namespace}")
    lines.append("")
    lines.append(f"#endif  // {guard}")
    lines.append("")
    return "\n".join(lines)


def render_source(cls: ClassModel, source_stem: str, header_name: str,
                  fragments: list[PortFragment],
                  method_port_prefix: str = "") -> str | None:
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
    for inc in _clean_includes(includes, method_port_prefix):
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

    # Original method access (public/private/protected), keyed by short name.
    access_by_name: dict[str, str | None] = {
        qn.rsplit("::", 1)[-1]: acc
        for qn, acc in get_class_methods_with_access(conn, class_qname)
    }
    # Port-dir prefix for THIS class's own method ports, e.g. "coot__molecule_t__".
    # Their phase-1 headers are dropped from includes (now in-class); other
    # ports' headers (util free-fns) are kept.
    method_port_prefix = sanitize_name(class_qname) + "__"

    # Header sees every fragment (all declarations + unioned members).
    header_text = render_header(cls, header_stem, all_frags, warnings,
                                access_by_name, method_port_prefix)
    header_path = out_dir / header_name
    header_path.write_text(header_text)

    # One .cc per source file.
    source_paths: list[Path] = []
    for src in ordered_files:
        src_stem = Path(src).stem
        cc_text = render_source(cls, src_stem, header_name, frags_by_file[src],
                                method_port_prefix)
        if cc_text is None:
            continue
        cc_path = out_dir / f"{src_stem}_gemmi.cc"
        cc_path.write_text(cc_text)
        source_paths.append(cc_path)

    # Normalize whitespace to coot's house style. The merge emits a sane
    # baseline indent, but only clang-format makes the spacing consistent across
    # the hand-concatenated pieces. Best-effort: a missing formatter is a
    # warning, not a failure.
    from .format import format_files
    _, fmt_err = format_files([header_path, *source_paths])
    if fmt_err:
        warnings.append(fmt_err)

    return MergeResult(
        header_path=header_path,
        source_paths=source_paths,
        warnings=warnings,
        extra_members=dict(cls.extra_members),
    )
