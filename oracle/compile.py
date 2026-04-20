"""Compilation helpers for oracle.cc programs."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .db import PROJECT_ROOT

# ── constants (mirrored from mmdb-refactor-ui/backend/config.py) ─────────────
CXX           = "c++"
COOT_API_DIR  = "/Users/dialpuri/lmb/build-coot-and-deps"
COOT_API_NAME = "cootapi"
MMDB_API_DIR  = "/opt/homebrew/Cellar/mmdb2/2.0.22/lib"
MMDB_API_NAME = "mmdb2"
GEMMI_INCLUDE = "/opt/homebrew/opt/gemmi/include"
CLIPPER_INCLUDE = "/opt/homebrew/Cellar/clipper4coot/2.1.20180802_3/include"
BOOST_INCLUDE = "/opt/homebrew/Cellar/boost/1.90.0_1/include"
MMDB_INCLUDE  = "/opt/homebrew/Cellar/mmdb2/2.0.22/include"
GSL_INCLUDE   = "/opt/homebrew/Cellar/gsl/2.8/include"
PNG_INCLUDE = "/opt/homebrew/Cellar/libpng/1.6.56/include"
GLM_INCLUDE = "/opt/homebrew/Cellar/glm/1.0.1/include"

def make_compile_cmd(oracle_cc: Path, output_bin: Path) -> str:
    includes = [PROJECT_ROOT, GEMMI_INCLUDE, CLIPPER_INCLUDE, BOOST_INCLUDE, MMDB_INCLUDE, GSL_INCLUDE, PNG_INCLUDE, GLM_INCLUDE]
    includes = " ".join(f'-I"{i}"' for i in includes)
    return (
        f'{CXX} -std=c++17 "{oracle_cc}" -o "{output_bin}" '
        f'{includes} '
        f'-pthread '
        f'-Wl,-rpath,{COOT_API_DIR} '
        f'-L "{COOT_API_DIR}" -l{COOT_API_NAME} '
        f'-L "{MMDB_API_DIR}" -l{MMDB_API_NAME}'
    )


def write_compile_script(out_dir: Path) -> Path:
    """Write compile.sh into out_dir and make it executable. Returns the path."""
    oracle_cc  = out_dir / "oracle.cc"
    output_bin = out_dir / "oracle"
    cmd = make_compile_cmd(oracle_cc, output_bin)

    script = out_dir / "compile.sh"
    script.write_text(f"#!/bin/sh\nset -e\n{cmd}\n")
    script.chmod(0o755)
    return script


def compile_oracle(out_dir: Path) -> tuple[bool, str]:
    """Run compile.sh in out_dir. Returns (success, output)."""
    script = out_dir / "compile.sh"
    if not script.exists():
        write_compile_script(out_dir)

    compile_log = out_dir / "compile.log"
    with open(compile_log, "w") as f:
        proc = subprocess.run(
            ["sh", str(script)],
            text=True,
            stdout=f,
            stderr=f,
            cwd=str(out_dir),
        )
    return proc.returncode == 0

