"""
Run one or more compiled oracles and print a summary.

Usage:
  python -m oracle.runner oracle-data/coot__molecule_t__cid_to_residue
  python -m oracle.runner oracle-data/          # all subdirs
"""
import argparse
import sys
from pathlib import Path

from .run import run_oracle
from .results import load_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compiled oracle binaries")
    parser.add_argument("paths", nargs="+", metavar="DIR",
                        help="Oracle output directory/directories (or a parent dir)")
    parser.add_argument("--reprint", action="store_true",
                        help="Print saved result.json instead of re-running")
    args = parser.parse_args()

    dirs: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if (p / "oracle" / "oracle").exists() or (p / "oracle" / "oracle.cc").exists():
            dirs.append(p)
        else:
            # Treat as parent — collect all child dirs that have an oracle binary.
            dirs.extend(sorted(d for d in p.iterdir()
                               if d.is_dir() and (d / "oracle" / "oracle").exists()))

    if not dirs:
        print("No compiled oracle binaries found.", file=sys.stderr)
        sys.exit(1)

    ok = fail = 0
    for d in dirs:
        name = d.name
        if args.reprint:
            result_path = d / "oracle" / "result.json"
            if not result_path.exists():
                print(f"{name}: no result.json")
                continue
            result = load_result(result_path)
        else:
            if not (d / "oracle" / "oracle").exists():
                print(f"{name}: not compiled")
                continue
            result = run_oracle(d / "oracle")

        status = "ok" if result.success else f"FAILED (exit {result.returncode})"
        print(f"{name}: {status}")
        for n, v in result.inputs.items():
            print(f"  INPUT  {n}: {v}")
        for n, v in result.outputs.items():
            print(f"  OUTPUT {n}: {v}")
        if not result.success and result.stderr.strip():
            print(f"  stderr: {result.stderr.strip()[:300]}")

        if result.success:
            ok += 1
        else:
            fail += 1

    print(f"\n{ok} ok  {fail} failed  ({len(dirs)} total)")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
