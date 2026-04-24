"""Compilation helpers for the combined gemmi port + test.

gemmi 0.7.x ships both a header tree and a shared library (libgemmi_cpp), so
the ported function can split across function.hh / function.cc if needed, and
we link against -lgemmi_cpp for symbols that live in the library.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..oracle.compile import (
    CXX, GEMMI_INCLUDE, BOOST_INCLUDE, GSL_INCLUDE, GLM_INCLUDE,
)
from ..test.compile import GTEST_INCLUDE, GTEST_LIB_DIR

GEMMI_LIB_DIR = "/opt/homebrew/opt/gemmi/lib"
GEMMI_LIB_NAME = "gemmi_cpp"

MAX_COMPILE_ATTEMPTS = 4


def make_gemmi_compile_cmd(
    test_cc: Path,
    output_bin: Path,
    function_cc: Path | None = None,
) -> str:
    """Compile test.cc (and optionally function.cc) linking against gemmi + gtest."""
    include_dirs = [str(test_cc.parent), GEMMI_INCLUDE, BOOST_INCLUDE,
                    GSL_INCLUDE, GLM_INCLUDE, GTEST_INCLUDE]
    includes = " ".join(f'-I"{i}"' for i in include_dirs)
    sources = f'"{test_cc.absolute()}"'
    if function_cc is not None and function_cc.exists():
        sources += f' "{function_cc.absolute()}"'
    return (
        f'{CXX} -std=c++17 {sources} -o "{output_bin.absolute()}" '
        f'{includes} -pthread '
        f'-Wl,-rpath,{GEMMI_LIB_DIR} '
        f'-L "{GEMMI_LIB_DIR}" -l{GEMMI_LIB_NAME} '
        f'-L "{GTEST_LIB_DIR}" -lgtest -lgtest_main'
    )


def compile_gemmi(
    test_cc: Path, output_bin: Path, function_cc: Path | None = None,
) -> tuple[bool, str]:
    cmd = make_gemmi_compile_cmd(test_cc, output_bin, function_cc)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          cwd=str(test_cc.parent))
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def run_gemmi_test_binary(test_bin: Path) -> tuple[bool, str]:
    proc = subprocess.run([str(test_bin.absolute())],
                          capture_output=True, text=True,
                          cwd=str(test_bin.parent))
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def write_compile_script(gemmi_subdir: Path, has_function_cc: bool) -> Path:
    test_cc     = gemmi_subdir / "test.cc"
    function_cc = gemmi_subdir / "function.cc" if has_function_cc else None
    test_bin    = gemmi_subdir / "test"
    cmd = make_gemmi_compile_cmd(test_cc, test_bin, function_cc)
    script = gemmi_subdir / "compile_gemmi.sh"
    script.write_text(f"#!/bin/sh\nset -e\n{cmd}\n")
    script.chmod(0o755)
    return script
