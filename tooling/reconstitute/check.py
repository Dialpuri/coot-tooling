"""clang-tidy lint gate for the merged `_gemmi` files.

The mechanical merge lint (`merge._clean_includes` / `merge._dedup_methods`)
only catches what a regex can see. The real defects in a merged class —
redefined helper classes, conflicting method bodies, unqualified members,
garbage declarations — are *semantic* and only a compiler frontend finds them.

clang-tidy runs the clang frontend, so enabling `clang-diagnostic-*` turns it
into a syntax/semantic gate (a superset of `clang++ -fsyntax-only`) while still
leaving room for real lint checks. We reuse the gemmi compile include dirs so
the coot tree + gemmi headers resolve, and a `--header-filter` limited to the
merged files keeps coot's own header warnings out of the report.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..gemmi.compile import (
    GEMMI_INCLUDE, BOOST_INCLUDE, GSL_INCLUDE, GLM_INCLUDE,
    CLIPPER_INCLUDE, RDKIT_INCLUDE,
)
from ..db import PROJECT_ROOT

# clang-diagnostic-* is the compiler frontend (errors == won't compile). The
# rest are high-signal correctness checks; style/modernize are deliberately
# omitted to keep the gate about *validity*, not taste.
DEFAULT_CHECKS = "-*,clang-diagnostic-*,bugprone-*,misc-definitions-in-headers"

_DIAG_RE = re.compile(r"^(?P<path>[^:]+):(?P<line>\d+):\d+:\s+(?P<level>error|warning):")


@dataclass
class FileReport:
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class CheckResult:
    reports: list[FileReport]                 # per-source (.cc) diagnostics only
    header_errors: list[str] = field(default_factory=list)    # shared, deduped
    header_warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.header_errors and all(r.ok for r in self.reports)

    @property
    def n_errors(self) -> int:
        return len(self.header_errors) + sum(len(r.errors) for r in self.reports)


def _include_args() -> list[str]:
    dirs = [PROJECT_ROOT, GEMMI_INCLUDE, BOOST_INCLUDE, GSL_INCLUDE,
            GLM_INCLUDE, CLIPPER_INCLUDE, RDKIT_INCLUDE]
    return [f"-I{d}" for d in dirs]


def clang_tidy_path() -> str | None:
    return shutil.which("clang-tidy")


def check_file(cc: Path, checks: str = DEFAULT_CHECKS) -> tuple[FileReport, list[str], list[str]]:
    """Run clang-tidy on one merged `.cc`.

    Returns (source_report, header_errors, header_warnings). The `.cc` includes
    the merged header, so header diagnostics surface in every compile — they're
    split out here so the caller can dedupe them into a single shared section
    instead of repeating them per file. The `--header-filter` confines reported
    headers to the staging dir, dropping noise from the transitively-included
    real coot/gemmi headers.
    """
    cc = cc.resolve()
    stem_dir = re.escape(str(cc.parent))
    cmd = [
        clang_tidy_path() or "clang-tidy",
        f"--checks={checks}",
        f"--header-filter={stem_dir}/.*\\.hh$",
        "--quiet",
        str(cc),
        "--",
        "-std=c++20",
        f"-I{cc.parent}",
        *_include_args(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    report = FileReport(path=cc)
    header_errors: list[str] = []
    header_warnings: list[str] = []
    staging = str(cc.parent)
    for line in (proc.stdout + proc.stderr).splitlines():
        m = _DIAG_RE.match(line)
        if not m:
            continue
        path = m.group("path")
        # Only diagnostics located in the staging dir are ours to fix.
        if not path.startswith(staging):
            continue
        is_err = m.group("level") == "error"
        if path.endswith(".hh"):           # shared header — caller dedupes
            (header_errors if is_err else header_warnings).append(line)
        else:
            (report.errors if is_err else report.warnings).append(line)
    return report, header_errors, header_warnings


def check_files(files: list[Path], checks: str = DEFAULT_CHECKS) -> CheckResult:
    reports: list[FileReport] = []
    h_errs: dict[str, None] = {}           # ordered, deduped
    h_warns: dict[str, None] = {}
    for f in files:
        rep, he, hw = check_file(f, checks)
        reports.append(rep)
        for line in he:
            h_errs.setdefault(line, None)
        for line in hw:
            h_warns.setdefault(line, None)
    return CheckResult(reports=reports,
                       header_errors=list(h_errs),
                       header_warnings=list(h_warns))
