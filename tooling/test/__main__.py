"""CLI: python -m tooling.test <oracle-dir> [--agent] [--model MODEL] [--verbose]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .generate import generate_test, DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Google Test from an oracle directory"
    )
    parser.add_argument("oracle_dir", type=Path, help="Path to oracle-data/<name>/")
    parser.add_argument("--model",   default=DEFAULT_MODEL, help="Ollama model to use")
    parser.add_argument("--agent",   action="store_true",
                        help="Agentic mode: model calls tools to verify headers and types")
    parser.add_argument("--verbose", action="store_true",
                        help="Print tool calls and thinking to the console")
    args = parser.parse_args()

    try:
        test_cc = generate_test(
            args.oracle_dir,
            model=args.model,
            agent=args.agent,
            verbose=args.verbose,
        )
        print(f"Generated: {test_cc}")
        print(f"Compile:   sh {test_cc.parent / 'compile_test.sh'}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


main()
