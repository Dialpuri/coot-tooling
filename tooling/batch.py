#!/usr/bin/env python3
"""
Full pipeline: generate oracles then tests for all methods in a class or file.

Usage (class mode):
  # Oracle + test for every method in coot::molecule_t
  python -m tooling.batch "coot::molecule_t"

  # Agentic oracle + agentic test, filtered to methods containing "cid"
  python -m tooling.batch "coot::molecule_t" --agent --filter cid

  # Skip oracle generation if oracle.cc already exists, only (re)generate tests
  python -m tooling.batch "coot::molecule_t" --skip-oracle

  # Parallel workers (each worker runs its own Ollama request)
  python -m tooling.batch "coot::molecule_t" --workers 4

  # List matching methods without generating anything
  python -m tooling.batch "coot::molecule_t" --list

Usage (file mode):
  # Oracle + test + gemmi for every function defined in a source file
  python -m tooling.batch_file src/coot/molecule.cc

  # Same flags as class mode apply
  python -m tooling.batch_file src/coot/molecule.cc --agent --workers 4
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .db import connect, get_class_functions, get_file_functions, get_internal_call_deps
from .oracle.generate import DEFAULT_MODEL, OUT_ROOT, generate_one, sanitize_name
from .test.generate import generate_test
from .gemmi.generate import generate_gemmi
from .gemmi.aggregate import aggregate_gemmi_files


# ── result tracking ───────────────────────────────────────────────────────────

class Result:
    def __init__(self, qname: str):
        self.qname      = qname
        self.skipped    = False
        self.oracle_ok: bool | None = None
        self.test_ok:   bool | None = None
        self.gemmi_ok:  bool | None = None
        self.error:     str  | None = None

    @property
    def short(self) -> str:
        return self.qname.rsplit("::", 1)[-1]


# ── dependency ordering ───────────────────────────────────────────────────────

def topo_order(deps: dict[str, set[str]]) -> list[str]:
    """Return qnames in bottom-up call order: callees (inside the batch)
    come before their callers. On cycles, break by picking the node with
    the fewest outstanding in-batch deps (deterministic tie-break: qname).
    """
    remaining = {q: set(d) for q, d in deps.items()}
    order: list[str] = []
    while remaining:
        ready = sorted(q for q, d in remaining.items() if not d)
        if not ready:
            pick = min(remaining, key=lambda q: (len(remaining[q]), q))
            ready = [pick]
        for q in ready:
            order.append(q)
            del remaining[q]
        for d in remaining.values():
            d.difference_update(ready)
    return order


# ── per-function worker ───────────────────────────────────────────────────────

def _process(
    qname: str,
    model: str,
    agent: bool,
    verbose: bool,
    skip_oracle: bool,
    skip_existing: bool,
    with_gemmi: bool = False,
) -> Result:
    r = Result(qname)
    out_dir = OUT_ROOT / sanitize_name(qname)
    oracle_cc = out_dir / "oracle" / "oracle.cc"

    # ── oracle phase ──────────────────────────────────────────────────────────
    if skip_oracle and oracle_cc.exists():
        r.oracle_ok = True  # treat pre-existing oracle as success
    elif skip_existing and oracle_cc.exists():
        r.skipped = True
        return r
    else:
        conn = connect()
        try:
            result_dir = generate_one(
                conn, qname, model=model, agent=agent, verbose=verbose,
            )
        except urllib.error.URLError as e:
            r.error = f"Ollama unreachable: {e}"
            return r
        finally:
            conn.close()

        if result_dir is None:
            r.error = "not found in DB"
            return r

        r.oracle_ok = True

    # ── test phase ────────────────────────────────────────────────────────────
    try:
        generate_test(
            out_dir, model=model, agent=agent, verbose=verbose,
        )
        r.test_ok = True
    except Exception as e:
        r.test_ok = False
        r.error = f"test generation failed: {e}"
        return r

    # ── gemmi port phase (optional) ───────────────────────────────────────────
    if with_gemmi:
        try:
            generate_gemmi(out_dir, qname, model=model, verbose=verbose)
            r.gemmi_ok = True
        except Exception as e:
            r.gemmi_ok = False
            r.error = f"gemmi port failed: {e}"

    return r


# ── summary ───────────────────────────────────────────────────────────────────

def _print_summary(results: list[Result]) -> None:
    sym = {True: "✓", False: "✗", None: " "}
    skip_sym = "–"

    has_gemmi = any(r.gemmi_ok is not None for r in results)
    header = (f"{'method':<50}  oracle  test"
              + ("  gemmi" if has_gemmi else ""))
    print("\n" + header)
    print("-" * len(header))

    ok = fail = skip = 0
    for r in sorted(results, key=lambda r: r.qname):
        if r.skipped:
            skip += 1
            print(f"{r.short:<50}  {skip_sym}")
            continue

        row = f"{r.short:<50}  {sym[r.oracle_ok]}       {sym[r.test_ok]}"
        if has_gemmi:
            row += f"      {sym[r.gemmi_ok]}"
        if r.error:
            row += f"  ← {r.error.splitlines()[0]}"
        print(row)

        stage_ok = r.oracle_ok and r.test_ok and (r.gemmi_ok is not False)
        if stage_ok:
            ok += 1
        else:
            fail += 1

    print(f"\n{ok} ok  {fail} failed  {skip} skipped  ({len(results)} total)")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate oracles + Google Tests for all methods in a class",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("class_name", help="Fully-qualified class name, e.g. coot::molecule_t")
    parser.add_argument("--filter",        metavar="STR",  help="Only process methods whose name contains STR")
    parser.add_argument("--mmdb-only",     action="store_true", help="Only process methods that use MMDB types (mmdb::*)")
    parser.add_argument("--model",         default=DEFAULT_MODEL)
    parser.add_argument("--agent",         action="store_true",  help="Agentic mode for both oracle and test generation")
    parser.add_argument("--verbose",       action="store_true", help="Print thinking and tool calls to console")
    parser.add_argument("--skip-oracle",   action="store_true", help="Skip oracle generation if oracle.cc already exists; go straight to test generation")
    parser.add_argument("--skip-existing", action="store_true", help="Skip methods that already have oracle.cc")
    parser.add_argument("--with-gemmi",    action="store_true",
                        help="After test succeeds, also run the combined gemmi port + test stage")
    parser.add_argument("--no-topo",       action="store_true",
                        help="Disable bottom-up call-graph ordering (default is enabled: "
                             "functions with no in-batch callees go first, so any callees "
                             "are already converted by the time their callers are processed)")
    parser.add_argument("--workers",       type=int, default=1, metavar="N",
                        help="Parallel workers (default 1)")
    parser.add_argument("--list",          action="store_true", help="List matching methods and exit")
    args = parser.parse_args()

    conn = connect()
    qnames = get_class_functions(conn, args.class_name, mmdb_only=args.mmdb_only)
    conn.close()

    if not qnames:
        print(f"No methods found for class: {args.class_name}", file=sys.stderr)
        sys.exit(1)

    if args.filter:
        qnames = [q for q in qnames if args.filter in q]
        if not qnames:
            print(f"No methods match filter '{args.filter}'", file=sys.stderr)
            sys.exit(1)

    # Bottom-up topological ordering: callees before callers. Skipped in
    # parallel mode because dependencies can't be enforced once workers run
    # concurrently — use --workers 1 to keep the guarantee.
    if not args.no_topo:
        if args.workers > 1:
            print("[warn] --workers > 1 disables call-graph ordering; "
                  "pass --no-topo to silence this.", file=sys.stderr)
        else:
            conn = connect()
            try:
                deps = get_internal_call_deps(conn, qnames)
            finally:
                conn.close()
            qnames = topo_order(deps)

    if args.list:
        for q in qnames:
            print(q)
        print(f"\n{len(qnames)} methods")
        return

    print(f"Processing {len(qnames)} methods from {args.class_name} "
          f"(model={args.model}, workers={args.workers}, agent={args.agent})")

    results: list[Result] = []

    if args.workers == 1:
        for i, qname in enumerate(qnames, 1):
            print(f"[{i}/{len(qnames)}] {qname.rsplit('::', 1)[-1]} ...", end=" ", flush=True)
            r = _process(qname, args.model, args.agent, args.verbose,
                         args.skip_oracle, args.skip_existing,
                         with_gemmi=args.with_gemmi)
            results.append(r)
            if r.skipped:
                print("skipped")
            elif r.error:
                print(f"FAILED — {r.error.splitlines()[0]}")
            else:
                print("ok")
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for qname in qnames:
                f = pool.submit(_process, qname, args.model, args.agent,
                                args.verbose, args.skip_oracle, args.skip_existing,
                                args.with_gemmi)
                futures[f] = qname
            for f in as_completed(futures):
                r = f.result()
                results.append(r)
                status = "skipped" if r.skipped else ("ok" if not r.error else "FAILED")
                print(f"  {r.short}: {status}")

    _print_summary(results)
    if any(not r.skipped and (not r.oracle_ok or not r.test_ok) for r in results):
        sys.exit(1)


def _aggregate(qnames: list[str], source_file: str, with_gemmi: bool) -> None:
    """Print aggregation results; called at the end of main_file."""
    if not with_gemmi:
        return
    hh, cc = aggregate_gemmi_files(qnames, source_file)
    print(f"\n[aggregate] {hh}")
    if cc:
        print(f"[aggregate] {cc}")


def main_file() -> None:
    parser = argparse.ArgumentParser(
        description="Run oracle + test + gemmi for every function defined in a source file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("file", help="Source file path (absolute, or a suffix of the stored path, e.g. src/coot/molecule.cc)")
    parser.add_argument("--filter",        metavar="STR",  help="Only process functions whose name contains STR")
    parser.add_argument("--model",         default=DEFAULT_MODEL)
    parser.add_argument("--agent",         action="store_true",  help="Agentic mode for oracle, test, and gemmi generation")
    parser.add_argument("--verbose",       action="store_true",  help="Print thinking and tool calls to console")
    parser.add_argument("--skip-oracle",   action="store_true",  help="Skip oracle generation if oracle.cc already exists")
    parser.add_argument("--skip-existing", action="store_true",  help="Skip functions that already have oracle.cc")
    parser.add_argument("--no-gemmi",      action="store_true",  help="Skip gemmi port stage (default: gemmi is run)")
    parser.add_argument("--no-topo",       action="store_true",  help="Disable bottom-up call-graph ordering")
    parser.add_argument("--workers",       type=int, default=1, metavar="N",
                        help="Parallel workers (default 1)")
    parser.add_argument("--list",          action="store_true",  help="List matching functions and exit")
    args = parser.parse_args()

    conn = connect()
    qnames = get_file_functions(conn, args.file)
    conn.close()

    if not qnames:
        print(f"No functions found for file: {args.file}", file=sys.stderr)
        sys.exit(1)

    if args.filter:
        qnames = [q for q in qnames if args.filter in q]
        if not qnames:
            print(f"No functions match filter '{args.filter}'", file=sys.stderr)
            sys.exit(1)

    if not args.no_topo:
        if args.workers > 1:
            print("[warn] --workers > 1 disables call-graph ordering; "
                  "pass --no-topo to silence this.", file=sys.stderr)
        else:
            conn = connect()
            try:
                deps = get_internal_call_deps(conn, qnames)
            finally:
                conn.close()
            qnames = topo_order(deps)

    if args.list:
        for q in qnames:
            print(q)
        print(f"\n{len(qnames)} functions")
        return

    with_gemmi = not args.no_gemmi
    print(f"Processing {len(qnames)} functions from {args.file} "
          f"(model={args.model}, workers={args.workers}, agent={args.agent}, gemmi={with_gemmi})")

    results: list[Result] = []

    if args.workers == 1:
        for i, qname in enumerate(qnames, 1):
            print(f"[{i}/{len(qnames)}] {qname.rsplit('::', 1)[-1]} ...", end=" ", flush=True)
            r = _process(qname, args.model, args.agent, args.verbose,
                         args.skip_oracle, args.skip_existing,
                         with_gemmi=with_gemmi)
            results.append(r)
            if r.skipped:
                print("skipped")
            elif r.error:
                print(f"FAILED — {r.error.splitlines()[0]}")
            else:
                print("ok")
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for qname in qnames:
                f = pool.submit(_process, qname, args.model, args.agent,
                                args.verbose, args.skip_oracle, args.skip_existing,
                                with_gemmi)
                futures[f] = qname
            for f in as_completed(futures):
                r = f.result()
                results.append(r)
                status = "skipped" if r.skipped else ("ok" if not r.error else "FAILED")
                print(f"  {r.short}: {status}")

    _print_summary(results)
    _aggregate(qnames, args.file, with_gemmi)
    if any(not r.skipped and (not r.oracle_ok or not r.test_ok) for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
