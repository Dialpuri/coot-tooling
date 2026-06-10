"""CLI for phase-2 reconstitution.

    python -m tooling.reconstitute coot::molecule_t [options]

Pipeline per class:
  gather passing phase-1 ports → lift each into a `<class>_gemmi` method
  (re-verified by the port's own test) → merge the verified pieces into
  `<stem>_gemmi.{hh,cc}` staging files.

By default it lifts every pending port then merges. Use --list to inspect, --no-lift
to merge what already exists, --no-merge to only lift.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..db import connect, get_class_functions, group_entries_by_source_file
from ..oracle.generate import OUT_ROOT, sanitize_name
from ..llm import OPENAI_HOSTS, set_openai_host
from ..ollama import OLLAMA_HOSTS, set_host
from .agent import DEFAULT_MODEL
from .agent_reexports import dep_extra_includes, dep_extra_sources
from .generate import lift_one, verify_source_port, SourcePortNotPassing
from .merge import merge_class

OPENAI_MODEL = "huggingface/Qwen3.6-27B"


def _port_dir(qname: str, sig_hash: str | None) -> Path:
    return OUT_ROOT / sanitize_name(qname, sig_hash)


def _has_phase1_port(port_dir: Path) -> bool:
    g = port_dir / "gemmi"
    return (g / "function.hh").exists() and (g / "test.cc").exists()


def _is_lifted(port_dir: Path) -> bool:
    return (port_dir / "reconstitute" / "reconstitute_ok").exists()


def _short(qname: str, sig_hash: str | None) -> str:
    name = qname.rsplit("::", 1)[-1]
    return f"{name} [{sig_hash}]" if sig_hash else name


# Internal states (4); "missing" and "broken" both display as "no-port" — a
# stale/non-compiling phase-1 port is not a usable input. They are still counted
# separately so the summary shows how many no-ports are broken vs never generated.
_DISPLAY = {"lifted": "lifted", "pending": "pending",
            "missing": "no-port", "broken": "no-port"}


def _verify_and_cache(qname: str, sig_hash: str | None, pd: Path) -> str:
    """Compile+run the phase-1 port, cache the verdict as a marker, return state."""
    conn = connect()  # fresh per call → thread-safe
    try:
        inc = dep_extra_includes(conn, qname)
        src = dep_extra_sources(conn, qname)
    finally:
        conn.close()
    ok, log = verify_source_port(pd / "gemmi", inc, src)
    recon = pd / "reconstitute"
    recon.mkdir(parents=True, exist_ok=True)
    broken = recon / "source_broken.log"
    okmark = recon / "source_ok"
    if ok:
        okmark.write_text("")
        if broken.exists():
            broken.unlink()
        return "pending"
    broken.write_text(log)
    if okmark.exists():
        okmark.unlink()
    return "broken"


def _status(qname: str, sig_hash: str | None, verify: bool = False) -> str:
    """Classify a port. With verify=True, (re)compile the source port for truth;
    otherwise use cached markers (fast) and assume 'pending' when unknown."""
    pd = _port_dir(qname, sig_hash)
    if not _has_phase1_port(pd):
        return "missing"
    if _is_lifted(pd):
        return "lifted"
    if verify:
        return _verify_and_cache(qname, sig_hash, pd)
    recon = pd / "reconstitute"
    if (recon / "source_broken.log").exists():
        return "broken"
    return "pending"


def _cmd_list(
    class_name: str,
    entries: list[tuple[str, str | None]],
    verify: bool = False,
    workers: int = 1,
) -> None:
    conn = connect()
    try:
        buckets = group_entries_by_source_file(conn, entries)
    finally:
        conn.close()

    # Precompute statuses (verify is the slow path → parallelise it).
    status: dict[tuple[str, str | None], str] = {}
    if verify and workers > 1:
        print(f"[verify] compiling {len(entries)} source port(s) with {workers} workers...")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_status, qn, sh, True): (qn, sh) for qn, sh in entries}
            for f in as_completed(futs):
                status[futs[f]] = f.result()
    else:
        if verify:
            print(f"[verify] compiling {len(entries)} source port(s)...")
        for qn, sh in entries:
            status[(qn, sh)] = _status(qn, sh, verify)

    counts = {"lifted": 0, "pending": 0, "missing": 0, "broken": 0}
    for src in sorted(buckets, key=lambda f: (-len(buckets[f]), f)):
        name = os.path.basename(src) if src else "<unresolved>"
        print(f"\n{name}  ({len(buckets[src])})")
        for qn, sh in buckets[src]:
            st = status[(qn, sh)]
            counts[st] += 1
            note = "  (gemmi port broken)" if st == "broken" else ""
            print(f"    {_DISPLAY[st]:8s}  {_short(qn, sh)}{note}")
    no_port = counts["missing"] + counts["broken"]
    print(f"\n{class_name}: "
          f"{counts['lifted']} lifted, {counts['pending']} pending, "
          f"{no_port} no-port ({counts['broken']} broken, {counts['missing']} never generated) "
          f"— {len(entries)} total"
          + ("" if verify else "   [run with --verify to detect broken ports]"))


def _lift_entries(
    entries: list[tuple[str, str | None]],
    class_name: str,
    model: str,
    workers: int,
    openai_hosts: list[str],
    ollama_hosts: list[str],
    force: bool,
    verbose: bool,
) -> dict[str, bool]:
    """Lift the given entries; return {label: passed}. Skips already-lifted unless force."""
    todo = []
    for qn, sh in entries:
        pd = _port_dir(qn, sh)
        if not _has_phase1_port(pd):
            continue
        if _is_lifted(pd) and not force:
            continue
        todo.append((qn, sh))
    if not todo:
        print("[lift] nothing to lift (all pending ports already lifted; use --force to redo)")
        return {}
    print(f"[lift] lifting {len(todo)} port(s) with {workers} worker(s)")

    # status: "pass" | "fail" | "skip" (broken source port)
    results: dict[str, str] = {}

    def _work(idx: int, qn: str, sh: str | None) -> tuple[str, str]:
        if openai_hosts:
            set_openai_host(openai_hosts[idx % len(openai_hosts)])
        set_host(ollama_hosts[idx % len(ollama_hosts)])
        label = _short(qn, sh)
        try:
            res = lift_one(_port_dir(qn, sh), class_name, qn, sig_hash=sh,
                           model=model, verbose=verbose)
            return label, ("pass" if res else "fail")
        except SourcePortNotPassing:
            print(f"[lift] {label}: SKIP — source gemmi port no longer compiles/passes")
            return label, "skip"
        except Exception as e:  # keep the batch going; record the failure
            print(f"[lift] {label}: ERROR {e}")
            return label, "fail"

    if workers == 1:
        for i, (qn, sh) in enumerate(todo):
            label, st = _work(i, qn, sh)
            results[label] = st
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_work, i, qn, sh) for i, (qn, sh) in enumerate(todo)]
            for f in as_completed(futs):
                label, st = f.result()
                results[label] = st
    npass = sum(1 for v in results.values() if v == "pass")
    nskip = sum(1 for v in results.values() if v == "skip")
    nfail = sum(1 for v in results.values() if v == "fail")
    print(f"[lift] {npass} passed, {nfail} failed, {nskip} skipped (broken source) "
          f"of {len(results)}")
    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m tooling.reconstitute",
        description="Reconstitute verified gemmi ports into a parallel <class>_gemmi class.",
    )
    p.add_argument("class_name", help="Fully-qualified class name, e.g. coot::molecule_t")
    p.add_argument("--filter", metavar="STR", help="Only methods whose qualified name contains STR")
    p.add_argument("--model", default=None,
                   help=f"Model id (default: {OPENAI_MODEL} for openai, else {DEFAULT_MODEL})")
    p.add_argument("--backend", default="openai", choices=["ollama", "openai"],
                   help="LLM backend (default: openai)")
    p.add_argument("--workers", type=int, default=1, metavar="N", help="Parallel lift workers")
    p.add_argument("--force", action="store_true", help="Re-lift ports that already passed")
    p.add_argument("--no-lift", action="store_false", dest="lift", default=True,
                   help="Skip lifting; merge whatever is already lifted")
    p.add_argument("--no-merge", action="store_false", dest="merge", default=True,
                   help="Lift only; do not merge")
    p.add_argument("--out-dir", default=None,
                   help="Merge output dir (default: generated-tests/_reconstituted)")
    p.add_argument("--list", action="store_true", help="List ports + lift status and exit")
    p.add_argument("--verify", action="store_true",
                   help="With --list: compile each phase-1 port to flag broken/stale "
                        "ones as no-port (slow; honours --workers and caches the verdict)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    conn = connect()
    entries = get_class_functions(conn, args.class_name, mmdb_only=True)
    conn.close()
    if not entries:
        print(f"No mmdb methods found for class: {args.class_name}", file=sys.stderr)
        sys.exit(1)
    if args.filter:
        entries = [(q, s) for (q, s) in entries if args.filter in q]
        if not entries:
            print(f"No methods match filter '{args.filter}'", file=sys.stderr)
            sys.exit(1)

    if args.list:
        _cmd_list(args.class_name, entries, verify=args.verify, workers=args.workers)
        return

    os.environ["CT_BACKEND"] = args.backend
    model = args.model or (OPENAI_MODEL if args.backend == "openai" else DEFAULT_MODEL)
    openai_hosts = OPENAI_HOSTS if args.backend == "openai" else []
    print(f"Reconstituting {args.class_name} "
          f"(backend={args.backend}, model={model}, workers={args.workers}, "
          f"lift={args.lift}, merge={args.merge})")

    if args.lift:
        _lift_entries(entries, args.class_name, model, args.workers,
                      openai_hosts, OLLAMA_HOSTS, args.force, args.verbose)

    if args.merge:
        conn = connect()
        try:
            out_dir = Path(args.out_dir) if args.out_dir else None
            res = merge_class(conn, args.class_name, out_dir=out_dir)
        finally:
            conn.close()
        print(f"\n[merge] header : {res.header_path}")
        for sp in res.source_paths:
            print(f"[merge] source : {sp}")
        if res.extra_members:
            print(f"[merge] extra members introduced (review): {res.extra_members}")
        if res.warnings:
            print(f"[merge] warnings ({len(res.warnings)}):")
            for w in res.warnings:
                print(f"    {w}")


if __name__ == "__main__":
    main()
