"""Compilation helpers for test.cc programs."""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..oracle.compile import (
    CXX, COOT_API_DIR, COOT_API_NAME, MMDB_API_DIR, MMDB_API_NAME,
    GEMMI_INCLUDE, CLIPPER_INCLUDE, BOOST_INCLUDE, MMDB_INCLUDE,
    GSL_INCLUDE, PNG_INCLUDE, GLM_INCLUDE,
)
from ..db import PROJECT_ROOT

GTEST_INCLUDE = "/opt/homebrew/include"
GTEST_LIB_DIR = "/opt/homebrew/lib"

MAX_COMPILE_ATTEMPTS = 3

CXX           = "c++"
COOT_API_DIR  = "/Users/dialpuri/lmb/build-coot-and-deps"
COOT_API_NAME = "cootapi"
MMDB_API_DIR  = "/opt/homebrew/Cellar/mmdb2/2.0.22/lib"
MMDB_API_NAME = "mmdb2"
CLIPPER_API_DIR  = "/opt/homebrew/Cellar/clipper4coot/2.1.20180802_3/lib"
GEMMI_INCLUDE = "/opt/homebrew/opt/gemmi/include"
CLIPPER_INCLUDE = "/opt/homebrew/Cellar/clipper4coot/2.1.20180802_3/include"
BOOST_INCLUDE = "/opt/homebrew/Cellar/boost/1.90.0_1/include"
MMDB_INCLUDE  = "/opt/homebrew/Cellar/mmdb2/2.0.22/include"
GSL_INCLUDE   = "/opt/homebrew/Cellar/gsl/2.8/include"
PNG_INCLUDE = "/opt/homebrew/Cellar/libpng/1.6.56/include"
GLM_INCLUDE = "/opt/homebrew/Cellar/glm/1.0.1/include"

def make_test_compile_cmd(test_cc: Path, output_bin: Path) -> str:
    includes = [PROJECT_ROOT, GEMMI_INCLUDE, CLIPPER_INCLUDE, BOOST_INCLUDE, MMDB_INCLUDE, GSL_INCLUDE, PNG_INCLUDE, GLM_INCLUDE, GTEST_INCLUDE]
    includes = " ".join(f'-I"{i}"' for i in includes)

    clipper_libraries = [
        "clipper-ccp4", "clipper-core", "clipper-cif", "clipper-cns", "clipper-contrib", "clipper-minimol", "clipper-mmdb",
        "clipper-phs"
    ]
    clipper_libraries = " ".join(f'-l{l}' for l in clipper_libraries)

    return (
        f'{CXX} -std=c++17 "{test_cc.absolute()}" -o "{output_bin.absolute()}" '
        f'{includes} '
        f'-pthread '
        f'-Wl,-rpath,{COOT_API_DIR} '
        f'-L "{COOT_API_DIR}" -l{COOT_API_NAME} '
        f'-L "{MMDB_API_DIR}" -l{MMDB_API_NAME} '
        f'-L "{CLIPPER_API_DIR}" {clipper_libraries} '
        f'-L "{GTEST_LIB_DIR}" -lgtest -lgtest_main '
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
