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
import queue
import sys
import traceback
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from .db import connect, get_class_functions, get_file_functions, get_internal_call_deps
from .oracle.generate import DEFAULT_MODEL, OUT_ROOT, generate_one, sanitize_name
from .test.generate import generate_test
from .gemmi.generate import generate_gemmi
from .gemmi.aggregate import aggregate_gemmi_files
from .ollama import OLLAMA_HOSTS, set_host, get_host


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


def topo_waves(deps: dict[str, set[str]]) -> list[list[str]]:
    """Return qnames grouped by topological level.

    Each wave contains qnames that depend ONLY on names in earlier waves —
    so they can be processed concurrently. Use this with ThreadPoolExecutor
    to keep the callees-first guarantee while still parallelising across
    workers. Cycle-breaking matches `topo_order`: when no deps are clear,
    promote the node with the fewest outstanding deps (deterministic).
    """
    remaining = {q: set(d) for q, d in deps.items()}
    waves: list[list[str]] = []
    while remaining:
        ready = sorted(q for q, d in remaining.items() if not d)
        if not ready:
            # Cycle: break it by promoting the node closest to ready, which
            # avoids stalling on a strongly-connected component.
            pick = min(remaining, key=lambda q: (len(remaining[q]), q))
            ready = [pick]
        waves.append(ready)
        for q in ready:
            del remaining[q]
        for d in remaining.values():
            d.difference_update(ready)
    return waves


# ── per-function worker ───────────────────────────────────────────────────────

def _is_complete(out_dir: Path) -> bool:
    """Return True if the function already has gemmi/function.hh and gemmi/test.cc."""
    return (out_dir / "gemmi" / "function.hh").exists() and \
           (out_dir / "gemmi" / "test.cc").exists()


def _test_is_passing(out_dir: Path) -> bool:
    """Return True if test/run.log exists and records an all-passed result."""
    log = out_dir / "test" / "run.log"
    return log.exists() and "[  PASSED  ]" in log.read_text()


def _process(
    qname: str,
    model: str,
    agent: bool,
    verbose: bool,
    skip_oracle: bool,
    skip_existing: bool,
    with_gemmi: bool = False,
    overwrite: bool = False,
) -> Result:
    r = Result(qname)
    out_dir = OUT_ROOT / sanitize_name(qname)
    oracle_cc = out_dir / "oracle" / "oracle.cc"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "batch.log"

    host = get_host()
    host_idx = OLLAMA_HOSTS.index(host) if host in OLLAMA_HOSTS else -1
    _COLORS = ["\033[36m", "\033[32m", "\033[33m", "\033[35m"]  # cyan, green, yellow, magenta
    _RESET  = "\033[0m"
    _color  = _COLORS[host_idx % len(_COLORS)] if host_idx >= 0 else ""

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(f"{_color}{line}{_RESET}", flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log(f"START {qname} [ollama #{host_idx} — {host}]")

    # ── completion check ──────────────────────────────────────────────────────
    if not overwrite and _is_complete(out_dir):
        log("SKIP — already complete (gemmi/function.hh + gemmi/test.cc exist)")
        r.skipped = True
        return r

    # ── oracle phase ──────────────────────────────────────────────────────────
    oracle_result_json = out_dir / "oracle" / "result.json"
    if not overwrite and oracle_result_json.exists():
        log("oracle: skipped (result.json exists)")
        r.oracle_ok = True
    elif skip_oracle and oracle_cc.exists():
        log("oracle: skipped (oracle.cc exists)")
        r.oracle_ok = True
    elif skip_existing and oracle_cc.exists():
        log("oracle: skipped (--skip-existing)")
        r.skipped = True
        return r
    else:
        log("oracle: generating ...")
        conn = connect()
        try:
            result_dir = generate_one(
                conn, qname, model=model, verbose=verbose,
            )
        except urllib.error.URLError as e:
            log(f"oracle: FAILED — Ollama unreachable: {e}")
            r.error = f"Ollama unreachable: {e}"
            return r
        except Exception as e:
            log(f"oracle: FAILED — {e}\n{traceback.format_exc()}")
            r.error = f"oracle failed: {e}"
            return r
        finally:
            conn.close()

        if result_dir is None:
            log("oracle: FAILED — not found in DB")
            r.error = "not found in DB"
            return r

        log("oracle: ok")
        r.oracle_ok = True

    # ── test phase ────────────────────────────────────────────────────────────
    if not overwrite and _test_is_passing(out_dir):
        log("test: skipped (already passing)")
        r.test_ok = True
    else:
        log("test: generating ...")
        try:
            generate_test(out_dir, model=model, agent=agent, verbose=verbose)
            log("test: ok")
            r.test_ok = True
        except Exception as e:
            log(f"test: FAILED — {e}\n{traceback.format_exc()}")
            r.test_ok = False
            r.error = f"test generation failed: {e}"
            return r

    # ── gemmi port phase (optional) ───────────────────────────────────────────
    if with_gemmi:
        log("gemmi: generating ...")
        try:
            generate_gemmi(out_dir, qname, model=model, verbose=verbose)
            log("gemmi: ok")
            r.gemmi_ok = True
        except Exception as e:
            log(f"gemmi: FAILED — {e}\n{traceback.format_exc()}")
            r.gemmi_ok = False
            r.error = f"gemmi port failed: {e}"

    log(f"DONE oracle={r.oracle_ok} test={r.test_ok} gemmi={r.gemmi_ok}")
    return r


# ── parallel scheduling ───────────────────────────────────────────────────────

def _run_in_parallel(qnames: list[str], args, *, label: str = "") -> list[Result]:
    """Submit every qname to a thread pool and collect Results.

    Use for one wave of a topo schedule, or for the whole batch when
    --no-topo is set. Results return in completion order, not submission
    order.
    """
    # Some entry points use --with-gemmi (default off), others use --no-gemmi
    # (default on). Resolve to a single bool here so the worker call is uniform.
    with_gemmi = (
        getattr(args, "with_gemmi", False)
        if hasattr(args, "with_gemmi")
        else not getattr(args, "no_gemmi", False)
    )

    hosts = getattr(args, "ollama_hosts", OLLAMA_HOSTS)
    host_queue: queue.Queue[str] = queue.Queue()
    for h in hosts:
        host_queue.put(h)

    def _process_with_host(qname: str) -> Result:
        host = host_queue.get()
        try:
            set_host(host)
            return _process(qname, args.model, args.agent, args.verbose,
                            args.skip_oracle, args.skip_existing,
                            with_gemmi, args.overwrite)
        finally:
            host_queue.put(host)

    out: list[Result] = []
    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for qname in qnames:
            f = pool.submit(_process_with_host, qname)
            futures[f] = qname
        for f in as_completed(futures):
            r = f.result()
            out.append(r)
            status = "skipped" if r.skipped else ("ok" if not r.error else "FAILED")
            prefix = f"  [{label}] " if label else "  "
            print(f"{prefix}{r.short}: {status}")
    return out


def _run_topo_waves(qnames: list[str], args) -> list[Result]:
    """Process qnames respecting call-graph deps with a single shared pool.

    A task is submitted as soon as all its in-batch deps have completed,
    rather than waiting for an entire wave barrier. This keeps both workers
    busy even when the dep graph is a long chain of size-1 waves.
    """
    conn = connect()
    try:
        deps = get_internal_call_deps(conn, qnames)
    finally:
        conn.close()

    with_gemmi = (
        getattr(args, "with_gemmi", False)
        if hasattr(args, "with_gemmi")
        else not getattr(args, "no_gemmi", False)
    )
    hosts = getattr(args, "ollama_hosts", OLLAMA_HOSTS)
    host_queue: queue.Queue[str] = queue.Queue()
    for h in hosts:
        host_queue.put(h)

    def _process_with_host(qname: str) -> Result:
        host = host_queue.get()
        try:
            set_host(host)
            return _process(qname, args.model, args.agent, args.verbose,
                            args.skip_oracle, args.skip_existing,
                            with_gemmi, args.overwrite)
        finally:
            host_queue.put(host)

    # pending[q] = set of in-batch deps not yet completed
    pending: dict[str, set[str]] = {q: set(deps.get(q, set())) for q in qnames}
    in_flight: dict[object, str] = {}  # future -> qname
    results: list[Result] = []

    def _submit_ready(pool) -> None:
        for q in list(pending):
            if not pending[q]:
                f = pool.submit(_process_with_host, q)
                in_flight[f] = q
                del pending[q]

    print(f"Dep-parallel schedule: {len(qnames)} qname(s), {len(deps)} with in-batch deps")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        _submit_ready(pool)
        while in_flight:
            done = next(as_completed(in_flight))
            qname = in_flight.pop(done)
            r = done.result()
            results.append(r)
            status = "skipped" if r.skipped else ("ok" if not r.error else "FAILED")
            print(f"  {r.short}: {status}", flush=True)
            # unlock any tasks that were waiting on this one
            for p in pending.values():
                p.discard(qname)
            _submit_ready(pool)

    return results


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
    parser.add_argument("--overwrite",     action="store_true",
                        help="Re-run all stages even if gemmi/function.hh + gemmi/test.cc already exist")
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

    # Bottom-up topological ordering: callees before callers.
    # Single-worker mode → flat order. Multi-worker mode → waves so parallelism
    # is preserved without violating the callees-first invariant.
    if not args.no_topo and args.workers == 1:
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

    hosts = getattr(args, "ollama_hosts", OLLAMA_HOSTS)
    if args.workers == 1:
        results: list[Result] = []
        set_host(hosts[0])
        for i, qname in enumerate(qnames, 1):
            print(f"[{i}/{len(qnames)}] {qname.rsplit('::', 1)[-1]}", flush=True)
            r = _process(qname, args.model, args.agent, args.verbose,
                         args.skip_oracle, args.skip_existing,
                         with_gemmi=args.with_gemmi, overwrite=args.overwrite)
            results.append(r)
    elif args.no_topo:
        results = _run_in_parallel(qnames, args)
    else:
        results = _run_topo_waves(qnames, args)

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
    parser.add_argument("--mmdb-only",     action="store_true", help="Only process functions that use MMDB types (mmdb::*)")
    parser.add_argument("--model",         default=DEFAULT_MODEL)
    parser.add_argument("--agent",         action="store_true",  help="Agentic mode for oracle, test, and gemmi generation")
    parser.add_argument("--verbose",       action="store_true",  help="Print thinking and tool calls to console")
    parser.add_argument("--skip-oracle",   action="store_true",  help="Skip oracle generation if oracle.cc already exists")
    parser.add_argument("--skip-existing", action="store_true",  help="Skip functions that already have oracle.cc")
    parser.add_argument("--no-gemmi",      action="store_true",  help="Skip gemmi port stage (default: gemmi is run)")
    parser.add_argument("--no-topo",       action="store_true",  help="Disable bottom-up call-graph ordering")
    parser.add_argument("--workers",       type=int, default=1, metavar="N",
                        help="Parallel workers (default 1)")
    parser.add_argument("--overwrite",     action="store_true",
                        help="Re-run all stages even if gemmi/function.hh + gemmi/test.cc already exist")
    parser.add_argument("--list",          action="store_true",  help="List matching functions and exit")
    args = parser.parse_args()

    conn = connect()
    qnames = get_file_functions(conn, args.file, mmdb_only=args.mmdb_only)
    conn.close()

    if not qnames:
        print(f"No functions found for file: {args.file}", file=sys.stderr)
        sys.exit(1)

    if args.filter:
        qnames = [q for q in qnames if args.filter in q]
        if not qnames:
            print(f"No functions match filter '{args.filter}'", file=sys.stderr)
            sys.exit(1)

    # Bottom-up topological ordering: callees before callers.
    # Single-worker mode → flat order. Multi-worker mode → waves so parallelism
    # is preserved without violating the callees-first invariant.
    if not args.no_topo and args.workers == 1:
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

    hosts = getattr(args, "ollama_hosts", OLLAMA_HOSTS)
    if args.workers == 1:
        results: list[Result] = []
        set_host(hosts[0])
        for i, qname in enumerate(qnames, 1):
            print(f"[{i}/{len(qnames)}] {qname.rsplit('::', 1)[-1]}", flush=True)
            r = _process(qname, args.model, args.agent, args.verbose,
                         args.skip_oracle, args.skip_existing,
                         with_gemmi=with_gemmi, overwrite=args.overwrite)
            results.append(r)
    elif args.no_topo:
        results = _run_in_parallel(qnames, args)
    else:
        results = _run_topo_waves(qnames, args)

    _print_summary(results)
    _aggregate(qnames, args.file, with_gemmi)
    if any(not r.skipped and (not r.oracle_ok or not r.test_ok) for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
