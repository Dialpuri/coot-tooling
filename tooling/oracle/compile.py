"""Compilation helpers for oracle.cc programs."""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..db import PROJECT_ROOT

# ── constants (mirrored from mmdb-refactor-ui/backend/config.py) ─────────────
CXX           = "c++"
COOT_API_DIR  = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/lib"
COOT_API_NAME = "coot-api"
MMDB_API_DIR  = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/lib"
MMDB_API_NAME = "mmdb2"
CLIPPER_API_DIR  = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/lib"
GEMMI_INCLUDE = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
CLIPPER_INCLUDE = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
BOOST_INCLUDE = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
MMDB_INCLUDE  = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
GSL_INCLUDE   = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
PNG_INCLUDE = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
GLM_INCLUDE = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"

def make_compile_cmd(oracle_cc: Path, output_bin: Path) -> str:
    includes = [PROJECT_ROOT, GEMMI_INCLUDE]
    includes = " ".join(f'-I"{i}"' for i in includes)

    clipper_libraries = [
        "clipper-ccp4", "clipper-core", "clipper-cif", "clipper-cns", "clipper-contrib", "clipper-minimol", "clipper-mmdb",
        "clipper-phs"
    ]
    clipper_libraries = " ".join(f'-l{l}' for l in clipper_libraries)

    return (
        f'{CXX} -std=c++17 "{oracle_cc}" -o "{output_bin}" '
        f'{includes}  '
        f'-pthread '
        f'-Wl,-rpath,{COOT_API_DIR} '
        f'-L "{COOT_API_DIR}" -l{COOT_API_NAME} '
        f'-l{MMDB_API_NAME} '
        f'{clipper_libraries}'

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

