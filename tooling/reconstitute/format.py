"""clang-format pass over the merged `_gemmi` files, using coot's own style.

The merge step assembles declarations and definitions by string concatenation,
prepending a fixed indent to each piece's first line. Multi-line decls (doc
comments, wrapped signatures) keep whatever indentation the lift agent emitted,
so the raw merged output has inconsistent spacing (mixed 3/4-space indents,
misaligned doc comments, namespace bodies not indented).

Rather than hand-roll a C++ pretty-printer, we run clang-format with coot's
`.clang-format` (the same one the rest of the tree is formatted to:
`BasedOnStyle: Google`, `IndentWidth: 3`, `NamespaceIndentation: All`). This
makes the reconstituted files match the house style exactly and lets merge.py
stop caring about whitespace. Note coot sets `SortIncludes: false`, so this does
NOT reorder includes — include hygiene stays the merge step's job.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..db import PROJECT_ROOT

# clang-format is on PATH after `module load compilers/llvm`; fall back to the
# known cluster build so the pass still runs in a bare shell.
_LLVM_FALLBACK = "/public/compilers/llvm-project-22.1.5/build/bin/clang-format"
_COOT_STYLE = Path(PROJECT_ROOT) / ".clang-format"


def clang_format_path() -> str | None:
    found = shutil.which("clang-format")
    if found:
        return found
    return _LLVM_FALLBACK if Path(_LLVM_FALLBACK).exists() else None


def _style_arg() -> str:
    """Point clang-format at coot's own .clang-format if present, else Google."""
    if _COOT_STYLE.exists():
        return f"--style=file:{_COOT_STYLE}"
    return "--style=Google"


def format_files(paths: list[Path]) -> tuple[list[Path], str | None]:
    """Format the given files in place with coot's style.

    Returns (formatted_paths, error). `error` is a human-readable string when
    clang-format is unavailable or fails — formatting is best-effort, so callers
    should warn and carry on rather than abort the merge.
    """
    paths = [p for p in paths if p and p.exists()]
    if not paths:
        return [], None
    cf = clang_format_path()
    if cf is None:
        return [], ("clang-format not found (PATH or the llvm module) — "
                    "merged files left unformatted")
    cmd = [cf, "-i", _style_arg(), *[str(p) for p in paths]]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return [], f"clang-format failed: {proc.stderr.strip() or proc.stdout.strip()}"
    return paths, None
