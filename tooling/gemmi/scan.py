"""Scan existing gemmi ports for redefinitions of real coot types.

A port that stubs out a real coot type (`struct simple_mesh_t { ... }`) instead
of including its header compiles in isolation but is fabricated — the stub lets
the body fake a result that games the frozen assertions, and it collides the
moment the port is reconstituted. This scanner reuses the SAME rule the agent
gate now enforces (`lint.redefined_coot_type_findings`) to report which of the
already-generated ports are affected, so a targeted re-run can be scoped to them
rather than re-running the whole corpus.

    python -m tooling.gemmi.scan                 # summary + per-port list
    python -m tooling.gemmi.scan --quiet         # just the port names (for xargs)
    python -m tooling.gemmi.scan --filter molecule_t
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from ..db import connect
from ..oracle.generate import OUT_ROOT
from .lint import coot_record_type_index, redefined_coot_type_findings

# A port's gemmi-stage C++ — the header and optional impl the agent emitted.
_PORT_FILES = ("function.hh", "function.cc")


def scan_port(port_dir: Path, type_index: dict[str, str]) -> dict[str, list[str]]:
    """Return {redefined_type: [findings]} for one port's gemmi files."""
    by_type: dict[str, list[str]] = {}
    for name in _PORT_FILES:
        f = port_dir / "gemmi" / name
        if not f.exists():
            continue
        for finding in redefined_coot_type_findings(
            f.read_text(errors="ignore"), type_index
        ):
            # Finding text is `line N: \`...\` redefines the real coot type
            # \`coot::<name>\` — ...`; key it by the type for the summary.
            key = finding.split("`coot::", 1)[1].split("`", 1)[0] \
                if "`coot::" in finding else "?"
            by_type.setdefault(key, []).append(f"{name}: {finding}")
    return by_type


def scan_all(filter_str: str | None = None) -> dict[str, dict[str, list[str]]]:
    """Scan every port under OUT_ROOT. Returns {port_name: {type: [findings]}}."""
    conn = connect()
    try:
        type_index = coot_record_type_index(conn)
    finally:
        conn.close()
    results: dict[str, dict[str, list[str]]] = {}
    for port_dir in sorted(OUT_ROOT.glob("coot__*")):
        if not port_dir.is_dir():
            continue
        if filter_str and filter_str not in port_dir.name:
            continue
        hits = scan_port(port_dir, type_index)
        if hits:
            results[port_dir.name] = hits
    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m tooling.gemmi.scan",
        description="List gemmi ports that redefine (stub) real coot types.",
    )
    p.add_argument("--filter", metavar="STR",
                   help="Only ports whose dir name contains STR")
    p.add_argument("--quiet", action="store_true",
                   help="Print only the offending port names (one per line)")
    p.add_argument("--verbose", action="store_true",
                   help="Print every finding line, not just the type names")
    args = p.parse_args(argv)

    results = scan_all(args.filter)

    if args.quiet:
        for port in results:
            print(port)
        return

    total_ports = sum(1 for _ in OUT_ROOT.glob("coot__*"))
    type_counts: Counter[str] = Counter()
    for hits in results.values():
        type_counts.update(hits.keys())

    for port, hits in results.items():
        print(f"\n{port}")
        for typ, findings in sorted(hits.items()):
            if args.verbose:
                for fnd in findings:
                    print(f"    {fnd}")
            else:
                print(f"    coot::{typ}")

    print(f"\n{'='*60}")
    print(f"{len(results)} of {total_ports} ports redefine a real coot type.")
    if type_counts:
        print("\nmost-stubbed coot types:")
        for typ, c in type_counts.most_common(20):
            print(f"  {c:4d}  coot::{typ}")
    print("\nRe-run just these with the redefinition rule now blocking, e.g.:")
    print("  python -m tooling.gemmi.scan --quiet | "
          "while read p; do python -m tooling.gemmi <port-args> --force; done")
    if results:
        sys.exit(1)   # non-zero so CI/precommit can gate on a clean corpus


if __name__ == "__main__":
    main()
