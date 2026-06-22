#!/usr/bin/env python3
"""
Generate an oracle.cc for a given function.

Usage:
  python -m tooling.generate "coot::molecule_t::get_bonds_mesh"
  python -m tooling.generate "coot::molecule_t::get_bonds_mesh" --model gemma4:31b
  python -m tooling.generate "coot::molecule_t::get_bonds_mesh" --dry-run

Outputs to:
  oracle-data/<sanitized_name>/prompt.txt
  oracle-data/<sanitized_name>/oracle.cc
"""
import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

from .runner import run_oracle, run_oracle_panel
from ..db import connect, get_function
from .render import build_oracle_prompt
from .agent import generate_with_agent, _revision_section
from .compile import write_compile_script, compile_oracle
from .notes import extract_oracle_notes, save_notes, summarise_behavior
from .coverage import compute_coverage, save_coverage, render_summary, render_for_prompt
from .pdb_selector import fixture_panel, pdb_path as make_pdb_path
from ..ollama import generate_url

OLLAMA_URL    = "http://localhost:11434/api/generate"  # kept for reference
DEFAULT_MODEL = "hugginface/Qwen3.6-27B"
OUT_ROOT      = Path(__file__).parent.parent.parent / "generated-tests"

CRITIQUE_INSTRUCTIONS = """\
You are reviewing a C++ oracle program that was generated to observe the inputs
and outputs of a specific function.

Critique the program below against the original context. Check for:
  - Incorrect or missing includes
  - Wrong construction of the receiver object
  - Methods or types used that are not shown in the context
  - Missing INPUT/OUTPUT print statements
  - Code that will not compile

If the program is correct and complete, respond with exactly: LGTM

If you can improve it, respond with the corrected program inside a ```cpp block,
with comments where you have changed it and why.\
"""


def sanitize_name(qname: str, sig_hash: str | None = None) -> str:
    """Convert a qualified name + optional overload hash to a safe directory name.

    Single-overload functions pass `sig_hash=None` and keep their legacy
    directory name (e.g. `coot__molecule_t__get_bonds_mesh/`). Overloaded
    functions pass a 6-char hex sig_hash so each overload gets its own
    directory (e.g. `coot__molecule_t__backrub_rotamer__461ac0/`).
    """
    base = re.sub(r"[^a-zA-Z0-9]", "_", qname).strip("_")
    if sig_hash:
        return f"{base}__{sig_hash}"
    return base


def find_function_dirs(qname: str, out_root: Path | None = None) -> list[Path]:
    """Return every per-overload output dir that exists for `qname`.

    Used by dep-resolution code (gemmi callee lookup, aggregation) that
    knows a callee qname but not which overload's port to load. Matches
    both the legacy single-overload form `<sanitized>/` and the per-overload
    `<sanitized>__<hash>/` form.
    """
    root = out_root if out_root is not None else OUT_ROOT
    base = sanitize_name(qname)
    out: list[Path] = []
    exact = root / base
    if exact.is_dir():
        out.append(exact)
    out.extend(sorted(root.glob(f"{base}__*")))
    return out


def call_ollama(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "think":  False,
        # "options": {"temperature": 0.2, "num_predict": 2048},
    }).encode()

    req = urllib.request.Request(
        generate_url(),
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip()


def extract_cpp(response: str) -> str:
    """Pull the C++ code block out of the LLM response if wrapped in markdown."""
    match = re.search(r"```(?:cpp|c\+\+)?\n(.*?)```", response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()


def critique_oracle(oracle_code: str, original_prompt: str, model: str) -> str | None:
    """Run a critique pass on oracle_code.

    Returns improved C++ source if the LLM found issues, or None if LGTM.
    """
    critique_prompt = (
        f"{CRITIQUE_INSTRUCTIONS}\n\n"
        f"--- ORIGINAL CONTEXT ---\n{original_prompt}\n"
        f"--- GENERATED PROGRAM ---\n```cpp\n{oracle_code}\n```\n"
    )
    response = call_ollama(critique_prompt, model)
    if response.strip().upper().startswith("LGTM"):
        return None
    return extract_cpp(response)


_REVISION_SNAPSHOT_FILES = (
    "oracle.cc", "result.json", "panel.json", "coverage.json",
    "agent_trace.txt", "prompt.txt",
)


def _has_mutation_blindspot(cov) -> bool:
    """True when the oracle observed a mutation that came back unchanged for
    every BEFORE/AFTER pair — the 'function did nothing' signal that usually
    means the oracle watched an invariant (atom count) instead of the field the
    function edits (coordinates, B-factors)."""
    d = cov.dynamic
    return d.n_before_after_pairs > 0 and d.n_identical_pairs == d.n_before_after_pairs


def _restore_snapshot(oracle_out: Path, snapshot: dict[str, str | None]) -> None:
    for name, content in snapshot.items():
        if content is not None:
            (oracle_out / name).write_text(content)


def _attempt_coverage_revision(
    conn, function_qname, sig_hash, model, oracle_out: Path,
    prior_oracle: str, prior_cov, fn_src, verbose: bool,
):
    """One bounded re-run of the oracle agent with coverage feedback steering it
    toward the right observable. Returns (oracle_code, panel, result, cov) of
    the revised oracle if it scores strictly better; otherwise restores the
    original artifacts and returns None."""
    feedback = _revision_section(prior_oracle, render_for_prompt(prior_cov))
    snapshot = {
        name: (oracle_out / name).read_text() if (oracle_out / name).exists() else None
        for name in _REVISION_SNAPSHOT_FILES
    }

    print(f"  [coverage] no-op mutation (score {prior_cov.score:.2f}) — "
          "one revision attempt with observable guidance...")
    new_code, _trace = generate_with_agent(
        conn, function_qname, model,
        oracle_out=oracle_out, verbose=verbose, sig_hash=sig_hash,
        revision_feedback=feedback,
    )
    if not new_code:
        _restore_snapshot(oracle_out, snapshot)
        return None
    (oracle_out / "oracle.cc").write_text(new_code)
    write_compile_script(oracle_out)
    compile_oracle(oracle_out)
    panel = run_oracle_panel(oracle_out)
    result = panel.primary()
    if not panel.success or result is None:
        _restore_snapshot(oracle_out, snapshot)
        return None
    cov = compute_coverage(fn_src["source_code"] if fn_src else "", result)
    if cov.score > prior_cov.score:
        save_coverage(cov, oracle_out / "coverage.json")
        print(f"  [coverage] revision improved score "
              f"{prior_cov.score:.2f} -> {cov.score:.2f}")
        return new_code, panel, result, cov
    print(f"  [coverage] revision did not improve "
          f"({cov.score:.2f} <= {prior_cov.score:.2f}); keeping original")
    _restore_snapshot(oracle_out, snapshot)
    return None


def generate_one(
    conn,
    function_qname: str,
    sig_hash: str | None = None,
    model: str = DEFAULT_MODEL,
    second_pass: bool = False,
    verbose: bool = False,
    out_root: Path = OUT_ROOT,
) -> Path | None:
    """Generate oracle.cc for a single function (overload).

    `sig_hash` disambiguates among overloaded names. Pass None for
    non-overloaded functions to keep the legacy unsuffixed output dir.

    Returns the output directory on success, None if the function wasn't found.
    Raises urllib.error.URLError if Ollama is unreachable.
    """
    out_dir = out_root / sanitize_name(function_qname, sig_hash)
    oracle_out = out_dir / "oracle"
    oracle_out.mkdir(parents=True, exist_ok=True)
    oracle_cc_path = oracle_out / "oracle.cc"

    # No per-function structure selection: the oracle reads its path from argv
    # and is verified across the whole fixture panel.
    # print(f"[pdb] oracle runs across the {len(fixture_panel())}-fixture panel "
        #   f"(generic navigation, no per-function structure)")

    oracle_code, trace = generate_with_agent(
        conn, function_qname, model,
        oracle_out=oracle_out, verbose=verbose,
        sig_hash=sig_hash,
    )
    (oracle_out / "agent_trace.txt").write_text(trace)
    if oracle_code is None:
        return None
    oracle_cc_path.write_text(oracle_code)

    # if second_pass:
    #     context = (oracle_out / "prompt.txt").read_text() if (oracle_out / "prompt.txt").exists() else ""
    #     improved = critique_oracle(oracle_code, context, model)
    #     if improved:
    #         (oracle_out / "oracle_second_pass.cc").write_text(improved)

    write_compile_script(oracle_out)
    compile_oracle(oracle_out)

    # Run the oracle across the multi-fixture panel (protein / ligand / RNA /
    # glycoprotein). Pass = at least one fixture produced observable output;
    # the per-fixture (input, output) pairs become the frozen assertions the
    # test stage checks, so a constant tuned to one structure can't pass.
    panel = run_oracle_panel(oracle_out)
    print(panel.summary())
    result = panel.primary()   # primary passing fixture, for coverage/notes

    # Coverage signal — heuristic check that the oracle did something
    # interesting. Persist for the test/gemmi stages to read.
    if panel.success and result is not None:
        try:
            fn_src = get_function(conn, function_qname, sig_hash)
            cov = compute_coverage(
                fn_src["source_code"] if fn_src else "",
                result,
            )
            save_coverage(cov, oracle_out / "coverage.json")
            print(f"  {render_summary(cov)}")
            for s in cov.signals:
                print(f"  [coverage] {s}")
            # Coverage-triggered revision: a no-op mutation almost always means
            # the oracle watched the wrong field. Give the agent one guided
            # retry, and adopt it only if it actually improves coverage. Updates
            # oracle_code/result so the notes/behavior passes use the better one.
            if _has_mutation_blindspot(cov):
                revised = _attempt_coverage_revision(
                    conn, function_qname, sig_hash, model, oracle_out,
                    oracle_code, cov, fn_src, verbose,
                )
                if revised:
                    oracle_code, panel, result, cov = revised
        except Exception as e:
            print(f"[coverage] skipped: {e}")

    # Extract structured notes from the working oracle for downstream stages.
    # Best-effort: a failure here should not fail oracle generation.
    if panel.success:
        try:
            notes = extract_oracle_notes(oracle_code, function_qname, model)
            # Behavior pass: document what the function does, grounded in the
            # original source + the oracle's actual output. Separate call from
            # the empirical notes pass (it interprets logic, they don't).
            if result is not None:
                try:
                    fn_src = get_function(conn, function_qname, sig_hash)
                    behavior = summarise_behavior(
                        fn_src["source_code"] if fn_src else "",
                        oracle_code,
                        result.summary(),
                        function_qname,
                        model,
                    )
                    if behavior:
                        notes = notes or {}
                        notes["behavior"] = behavior
                except Exception as e:
                    print(f"[behavior] summary skipped: {e}")
            if notes:
                save_notes(notes, oracle_out / "notes.json")
        except Exception as e:
            print(f"[notes] extraction skipped: {e}")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an oracle.cc for a function")
    parser.add_argument("function", help="Fully-qualified function name")
    parser.add_argument("--sig",         metavar="HASH", default=None,
                        help="Overload sig_hash (6-char hex). Required when the "
                             "function is overloaded; defaults to the first overload "
                             "the DB returns otherwise.")
    parser.add_argument("--model",       default=DEFAULT_MODEL, help="Ollama model")
    parser.add_argument("--backend",     default="openai", choices=["ollama", "openai"],
                        help="LLM backend (default: ollama)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable reasoning/thinking output (sets CT_THINK=0)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print the prompt without calling the LLM")
    parser.add_argument("--second-pass", action="store_true",
                        help="Critique and optionally improve the generated oracle")
    parser.add_argument("--verbose",     action="store_true",
                        help="Print thinking and tool calls to the console")
    args = parser.parse_args()

    os.environ["CT_BACKEND"] = args.backend
    if args.no_thinking:
        os.environ["CT_THINK"] = "0"

    conn = connect()

    if args.dry_run:
        prompt = build_oracle_prompt(conn, args.function, sig_hash=args.sig)
        conn.close()
        if prompt is None:
            print(f"Function not found in DB: {args.function}", file=sys.stderr)
            sys.exit(1)
        print(prompt)
        return

    print(f"Calling {args.model}... (agent mode)")
    try:
        out_dir = generate_one(
            conn, args.function,
            sig_hash=args.sig,
            model=args.model,
            second_pass=args.second_pass,
            verbose=args.verbose,
        )
    except urllib.error.URLError as e:
        print(f"Ollama not reachable: {e}\nStart it with: ollama serve", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()

    if out_dir is None:
        print(f"Function not found in DB: {args.function}", file=sys.stderr)
        sys.exit(1)

    # for f in sorted(out_dir.iterdir()):
    #     print(f"Saved → {f}")


if __name__ == "__main__":
    main()
