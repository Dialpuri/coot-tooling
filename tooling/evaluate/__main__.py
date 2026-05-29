"""CLI for tooling.evaluate.

Examples:

  # Evaluate a single function dir (auto-detects first failing stage):
  python -m tooling.evaluate generated-tests/coot__util__number_of_residues_in_molecule

  # Force a specific stage:
  python -m tooling.evaluate generated-tests/coot__... --stage gemmi

  # Evaluate every function dir under generated-tests/, parallel, resumable:
  python -m tooling.evaluate generated-tests/ --all --workers 4 --skip-existing
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .detect import first_failing_stage, stage_statuses, STAGES
from .evaluator import evaluate_trace, EvaluationResult, DEFAULT_MODEL
from ..ollama import OLLAMA_HOSTS, set_host
from ..llm import OPENAI_HOSTS, set_openai_host


def _backend_is_openai() -> bool:
    return os.environ.get("CT_BACKEND", "ollama").lower() == "openai"


def _set_worker_host(h: str) -> None:
    """Bind this worker thread to host `h` for the active backend."""
    if _backend_is_openai():
        set_openai_host(h)
    else:
        set_host(h)


def _qname_from_dir(d: Path) -> str:
    return d.name.replace("__", "::")


def _is_function_dir(d: Path) -> bool:
    return d.is_dir() and any((d / s / "agent_trace.txt").exists() for s in STAGES)


def _resolve_stage(func_dir: Path, stage: str | None) -> tuple[str, str] | None:
    """Return (stage_name, detection_reason) or None if nothing to evaluate."""
    if stage is None:
        s = first_failing_stage(func_dir)
        if s is None:
            return None
        return s.name, s.reason
    all_statuses = {x.name: x for x in stage_statuses(func_dir)}
    if stage not in all_statuses or not all_statuses[stage].present:
        return None
    return stage, all_statuses[stage].reason


def _evaluate_one(func_dir: Path, *, stage: str | None,
                  model: str, write: bool,
                  skip_existing: bool = False,
                  redo_ran_out_of_turns: bool = False,
                  quiet: bool = False) -> EvaluationResult | None:
    def _say(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    qname = _qname_from_dir(func_dir)
    resolved = _resolve_stage(func_dir, stage)
    if resolved is None:
        _say(f"[skip] {func_dir.name}: no failing stage attempted")
        return None
    stage_name, reason = resolved

    out_file = func_dir / "evaluate" / f"{stage_name}.json"
    if skip_existing and out_file.exists():
        if redo_ran_out_of_turns:
            try:
                existing = json.loads(out_file.read_text())
                if existing.get("failure_mode") == "ran_out_of_turns":
                    _say(f"[redo] {func_dir.name}: re-evaluating ran_out_of_turns result")
                else:
                    _say(f"[skip] {func_dir.name}: {out_file.name} exists")
                    return None
            except Exception:
                pass
        else:
            _say(f"[skip] {func_dir.name}: {out_file.name} exists")
            return None

    trace_path = func_dir / stage_name / "agent_trace.txt"
    if not trace_path.exists():
        _say(f"[skip] {func_dir.name}: no {stage_name}/agent_trace.txt")
        return None

    _say(f"[eval] {func_dir.name} :: {stage_name} ({reason})")
    try:
        result = evaluate_trace(
            qname=qname,
            stage=stage_name,
            detection_reason=reason,
            trace_path=trace_path,
            model=model,
        )
    except Exception as e:
        msg = str(e)
        if ("exceed_context_size_error" in msg
                or "exceeds the available context size" in msg):
            from ..batch import _record_context_size_skip
            note = _record_context_size_skip(qname, None, "evaluate", e)
            _say(f"[skip] {func_dir.name}: context size exceeded — recorded in {note}")
            return None
        raise

    if write:
        out_file.parent.mkdir(exist_ok=True)
        out_file.write_text(json.dumps(result.to_dict(), indent=2))

    _say(
        f"   → {result.failure_mode} (confidence={result.confidence})\n"
        f"     {result.note}"
    )
    return result


def _evaluate_many(dirs: list[Path], *, stage: str | None, model: str,
                   write: bool, skip_existing: bool, redo_ran_out_of_turns: bool,
                   workers: int, hosts: list[str],
                   progress: bool = False) -> list[EvaluationResult]:
    """Run _evaluate_one across dirs with a thread pool, one Ollama host per worker."""
    if progress:
        from tqdm import tqdm
        bar = tqdm(total=len(dirs), desc="evaluate", unit="fn")
    else:
        bar = None

    def _advance() -> None:
        if bar is not None:
            bar.update(1)

    if workers <= 1:
        results: list[EvaluationResult] = []
        if hosts:
            _set_worker_host(hosts[0])
        for d in dirs:
            r = _evaluate_one(d, stage=stage, model=model,
                              write=write, skip_existing=skip_existing,
                              redo_ran_out_of_turns=redo_ran_out_of_turns,
                              quiet=progress)
            _advance()
            if r is not None:
                results.append(r)
        if bar is not None:
            bar.close()
        return results

    host_queue: queue.Queue[str] = queue.Queue()
    for h in hosts:
        host_queue.put(h)

    def _task(d: Path) -> EvaluationResult | None:
        h = host_queue.get()
        try:
            _set_worker_host(h)
            return _evaluate_one(d, stage=stage, model=model,
                                 write=write, skip_existing=skip_existing,
                                 redo_ran_out_of_turns=redo_ran_out_of_turns,
                                 quiet=progress)
        finally:
            host_queue.put(h)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, d): d for d in dirs}
        for f in as_completed(futures):
            try:
                r = f.result()
            except Exception as e:
                if not progress:
                    print(f"[error] {futures[f].name}: {e}", flush=True)
            else:
                if r is not None:
                    results.append(r)
            _advance()
    if bar is not None:
        bar.close()
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tooling.evaluate")
    ap.add_argument("path", type=Path,
                    help="A generated-tests/<func>/ dir, or generated-tests/ with --all.")
    ap.add_argument("--stage", choices=STAGES, default=None,
                    help="Force evaluation of this stage instead of the first failing one.")
    ap.add_argument("--all", action="store_true",
                    help="Treat `path` as a parent dir and evaluate every function subdir.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"LLM model name (default: {DEFAULT_MODEL}).")
    ap.add_argument("--no-write", action="store_true",
                    help="Do not write evaluate/<stage>.json next to the trace.")
    ap.add_argument("--summary", type=Path, default=None,
                    help="With --all: write an aggregate JSON to this path.")
    ap.add_argument("--workers", type=int, default=1, metavar="N",
                    help="Parallel workers when running with --all (default 1). "
                         "Hosts are round-robin assigned from OLLAMA_HOSTS.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip dirs whose evaluate/<stage>.json already exists.")
    ap.add_argument("--redo-ran-out-of-turns", action="store_true",
                    help="With --skip-existing: re-evaluate any result previously "
                         "classified as ran_out_of_turns instead of skipping it.")
    ap.add_argument("--progress", action="store_true",
                    help="Suppress per-item output and show only a tqdm progress bar "
                         "(requires --all).")
    args = ap.parse_args(argv)

    if not args.path.exists():
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 2

    if _backend_is_openai():
        hosts = list(OPENAI_HOSTS)
        hosts_var = "OPENAI_HOSTS"
    else:
        hosts = list(OLLAMA_HOSTS)
        hosts_var = "OLLAMA_HOSTS"
    if args.workers > 1 and len(hosts) < args.workers:
        # OK to oversubscribe a host; just warn so the user knows.
        print(f"note: {args.workers} workers but only {len(hosts)} {hosts_var} — "
              "workers will share hosts.", file=sys.stderr)
        # Pad the queue so every worker can checkout a host slot.
        hosts = (hosts * ((args.workers // len(hosts)) + 1))[:args.workers]

    if args.all:
        dirs = sorted(d for d in args.path.iterdir() if _is_function_dir(d))
        if not dirs:
            print(f"error: no function dirs found under {args.path}", file=sys.stderr)
            return 2
        results = _evaluate_many(
            dirs, stage=args.stage, model=args.model,
            write=not args.no_write, skip_existing=args.skip_existing,
            redo_ran_out_of_turns=args.redo_ran_out_of_turns,
            workers=args.workers, hosts=hosts, progress=args.progress,
        )
        from collections import Counter
        tally = Counter(r.failure_mode for r in results)
        by_stage: dict[str, Counter] = {}
        for r in results:
            by_stage.setdefault(r.stage, Counter())[r.failure_mode] += 1
        print("\n=== Failure-mode tally (overall) ===")
        for mode, n in tally.most_common():
            print(f"  {n:4d}  {mode}")
        for stage_name in STAGES:
            if stage_name in by_stage:
                print(f"\n=== {stage_name} ===")
                for mode, n in by_stage[stage_name].most_common():
                    print(f"  {n:4d}  {mode}")
        if args.summary:
            args.summary.write_text(json.dumps(
                {"results": [r.to_dict() for r in results],
                 "tally": dict(tally),
                 "tally_by_stage": {k: dict(v) for k, v in by_stage.items()}},
                indent=2,
            ))
            print(f"\nwrote {args.summary}")
        return 0

    if not _is_function_dir(args.path):
        print(f"error: {args.path} does not look like a function dir "
              "(no */agent_trace.txt). Did you mean --all?", file=sys.stderr)
        return 2
    if hosts:
        set_host(hosts[0])
    r = _evaluate_one(args.path, stage=args.stage, model=args.model,
                      write=not args.no_write,
                      skip_existing=args.skip_existing,
                      redo_ran_out_of_turns=args.redo_ran_out_of_turns)
    return 0 if r is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
