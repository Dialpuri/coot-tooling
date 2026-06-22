"""Compute the cross-unit dependency closure for reconstituting a class.

A lifted `<class>_gemmi` method may call a function that lives in *another*
namespace or class (`coot::util::get_residue_gemmi`, `coot::primitive_chi_angles`
methods, …). Its merged fragment still `#include`s that callee's phase-1 port
header out of `generated-tests/`, so the class cannot be committed into the coot
tree until those callee units are themselves reconstituted.

This module finds exactly that set — the transitive closure of non-self port
units that the class's *passing* fragments depend on — and classifies each unit
so the orchestrator can reconstitute it the right way:

  * free-function units (`coot::util`, `coot::co`, …)  -> deterministic merge,
    no LLM lift (the phase-1 port is already a free function in final shape);
  * class units (`coot::primitive_chi_angles`)         -> the ClassModel lift,
    and the dependent methods that call them must be re-lifted.

The closure is read off the filesystem (which port headers a fragment includes),
then each unit is resolved back to a DB function for its container + source file.
"""
from __future__ import annotations

import re
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..oracle.generate import OUT_ROOT, sanitize_name
from ..db import (
    connect, get_class_functions, get_containing_class,
    get_function_source_file, _per_overload_entries,
)

# `#include "...generated-tests/<unit>/gemmi/function.hh"` — captures the port
# directory name (the sanitized callee qname, possibly with a sig-hash suffix).
_PORT_INCLUDE = re.compile(
    r'#\s*include\s*"[^"]*generated-tests/([A-Za-z0-9_]+)/gemmi/function\.hh"')


def _fragment_port_units(function_hh: Path) -> set[str]:
    """Port-dir names a reconstitute (or phase-1) fragment header includes."""
    units: set[str] = set()
    if not function_hh.exists():
        return units
    for line in function_hh.read_text(errors="ignore").splitlines():
        m = _PORT_INCLUDE.search(line)
        if m:
            units.add(m.group(1))
    return units


def _seed_units(class_qname: str) -> set[str]:
    """Non-self port units that PASSING reconstitute fragments of the class include."""
    self_prefix = sanitize_name(class_qname) + "__"
    seed: set[str] = set()
    for recon in OUT_ROOT.glob(f"{sanitize_name(class_qname)}__*/reconstitute"):
        if not (recon / "reconstitute_ok").exists():
            continue
        for unit in _fragment_port_units(recon / "function.hh"):
            if not unit.startswith(self_prefix):
                seed.add(unit)
    return seed


def closure_unit_dirs(class_qname: str) -> set[str]:
    """Transitive set of non-self port units the class's passing ports depend on.

    Seeds from the passing fragments, then follows each unit's own phase-1
    `gemmi/function.hh` port includes (a util fn may call another util fn).
    """
    self_prefix = sanitize_name(class_qname) + "__"
    closure: set[str] = set()
    q: deque[str] = deque(_seed_units(class_qname))
    while q:
        unit = q.popleft()
        if unit in closure:
            continue
        closure.add(unit)
        for dep in _fragment_port_units(OUT_ROOT / unit / "gemmi" / "function.hh"):
            if not dep.startswith(self_prefix) and dep not in closure:
                q.append(dep)
    return closure


# ── resolving a port-dir name back to a DB function ───────────────────────────

def _build_dir_index(conn: sqlite3.Connection) -> dict[str, tuple[str, str | None]]:
    """Map every function's port-dir name -> (qname, sig_hash).

    Mirrors how phase-1 named its output dirs (`sanitize_name(qname, sig_hash)`),
    so a `generated-tests/<dir>` name resolves unambiguously back to the DB row —
    more robust than reversing the `::`→`__` substitution by hand.
    """
    rows = conn.execute("""
        SELECT DISTINCT qualified_name, display_name, line_start
        FROM functions
        WHERE kind IN ('CXX_METHOD', 'CONSTRUCTOR', 'DESTRUCTOR',
                       'FUNCTION_TEMPLATE', 'FUNCTION_DECL')
        ORDER BY line_start
    """).fetchall()
    index: dict[str, tuple[str, str | None]] = {}
    for qn, sh in _per_overload_entries(rows):
        index.setdefault(sanitize_name(qn, sh), (qn, sh))
        # Also register the hash-less form: a legacy port dir was named before
        # the overload became ambiguous, so `coot__util__get_residue` must still
        # resolve even though the DB now keys it with a sig-hash. First overload
        # wins — container/source classification is the same across overloads.
        index.setdefault(sanitize_name(qn), (qn, sh))
    return index


@dataclass
class ResolvedUnit:
    unit_dir: str                 # "coot__util__get_residue"
    qname: str | None             # "coot::util::get_residue" (None if unresolved)
    sig_hash: str | None
    container: str | None         # "coot::util" / "coot::primitive_chi_angles"
    is_class: bool                # True -> ClassModel lift; False -> free-fn merge
    source_file: str | None       # defining .cc/.hh


@dataclass
class ClosureReport:
    class_qname: str
    units: list[ResolvedUnit] = field(default_factory=list)

    @property
    def free_function_units(self) -> list[ResolvedUnit]:
        return [u for u in self.units if not u.is_class]

    @property
    def class_units(self) -> list[ResolvedUnit]:
        return [u for u in self.units if u.is_class]

    @property
    def unresolved(self) -> list[ResolvedUnit]:
        return [u for u in self.units if u.qname is None]


def reconstitution_closure(
    conn: sqlite3.Connection, class_qname: str,
) -> ClosureReport:
    """The classified dependency closure that must be reconstituted before `class_qname`."""
    index = _build_dir_index(conn)
    report = ClosureReport(class_qname=class_qname)
    for unit in sorted(closure_unit_dirs(class_qname)):
        qn, sh = index.get(unit, (None, None))
        container = None
        is_class = False
        source = None
        if qn is not None:
            source = get_function_source_file(conn, qn, sh)
            if "::" in qn:
                container = qn.rsplit("::", 1)[0]
                parent = get_containing_class(conn, qn)
                # A `types` row whose kind is a record => the container is a class
                # (so it needs the stateful ClassModel lift). No row, or a
                # namespace kind => free functions in a namespace.
                is_class = bool(parent) and parent["kind"] in (
                    "CLASS_DECL", "STRUCT_DECL", "CLASS_TEMPLATE",
                    "class", "struct",
                )
        report.units.append(ResolvedUnit(
            unit_dir=unit, qname=qn, sig_hash=sh, container=container,
            is_class=is_class, source_file=source,
        ))
    return report


if __name__ == "__main__":
    import sys
    cls = sys.argv[1] if len(sys.argv) > 1 else "coot::molecule_t"
    conn = connect()
    try:
        rep = reconstitution_closure(conn, cls)
    finally:
        conn.close()
    print(f"Dependency closure for {cls}: {len(rep.units)} unit(s)\n")
    print(f"Free-function units ({len(rep.free_function_units)}) — deterministic merge, no lift:")
    for u in rep.free_function_units:
        src = Path(u.source_file).name if u.source_file else "?"
        print(f"    {u.container or '??':28s}  {u.qname or u.unit_dir}   [{src}]")
    print(f"\nClass units ({len(rep.class_units)}) — ClassModel lift, dependents re-lifted:")
    for u in rep.class_units:
        src = Path(u.source_file).name if u.source_file else "?"
        print(f"    {u.container or '??':28s}  {u.qname or u.unit_dir}   [{src}]")
    if rep.unresolved:
        print(f"\nUnresolved ({len(rep.unresolved)}):")
        for u in rep.unresolved:
            print(f"    {u.unit_dir}")
