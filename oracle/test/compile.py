"""Compilation helpers for test.cc programs."""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..compile import (
    CXX, COOT_API_DIR, COOT_API_NAME, MMDB_API_DIR, MMDB_API_NAME,
    GEMMI_INCLUDE, CLIPPER_INCLUDE, BOOST_INCLUDE, MMDB_INCLUDE,
    GSL_INCLUDE, PNG_INCLUDE, GLM_INCLUDE,
)
from ..db import PROJECT_ROOT

GTEST_INCLUDE = "/opt/homebrew/include"
GTEST_LIB_DIR = "/opt/homebrew/lib"

MAX_COMPILE_ATTEMPTS = 3


def make_test_compile_cmd(test_cc: Path, output_bin: Path) -> str:
    includes = [
        PROJECT_ROOT, GEMMI_INCLUDE, CLIPPER_INCLUDE, BOOST_INCLUDE,
        MMDB_INCLUDE, GSL_INCLUDE, PNG_INCLUDE, GLM_INCLUDE, GTEST_INCLUDE,
    ]
    inc_flags = " ".join(f'-I"{i}"' for i in includes)
    return (
        f'{CXX} -std=c++17 "{test_cc.absolute()}" -o "{output_bin.absolute()}" '
        f'{inc_flags} '
        f'-pthread '
        f'-Wl,-rpath,{COOT_API_DIR} '
        f'-L "{COOT_API_DIR}" -l{COOT_API_NAME} '
        f'-L "{MMDB_API_DIR}" -l{MMDB_API_NAME} '
        f'-L "{GTEST_LIB_DIR}" -lgtest -lgtest_main'
    )


def compile_test_cc(test_cc: Path, output_bin: Path) -> tuple[bool, str]:
    """Compile test_cc. Returns (success, compiler output)."""
    cmd = make_test_compile_cmd(test_cc, output_bin)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          cwd=str(test_cc.parent))
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def run_test_binary(test_bin: Path) -> tuple[bool, str]:
    """Run a compiled test binary. Returns (all tests passed, output)."""
    proc = subprocess.run(
        [str(test_bin.absolute())],
        capture_output=True, text=True,
        cwd=str(test_bin.parent),
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def write_compile_script(test_subdir: Path) -> Path:
    """Write compile_test.sh into test_subdir and make it executable."""
    test_cc  = test_subdir / "test.cc"
    test_bin = test_subdir / "test"
    cmd = make_test_compile_cmd(test_cc, test_bin)
    script = test_subdir / "compile_test.sh"
    script.write_text(f"#!/bin/sh\nset -e\n{cmd}\n")
    script.chmod(0o755)
    return script
