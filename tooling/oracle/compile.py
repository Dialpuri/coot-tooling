"""Compilation helpers for oracle.cc programs."""
from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

from ..db import PROJECT_ROOT

# CCP4 setup script — sourcing it exports CLIBD/CLIBD_MON/SYMINFO/etc. that
# coot needs at *runtime* (protein_geometry::init_standard, map symmetry, ...).
CCP4_SETUP = "/public/xtal/ccp4/ccp4-64/ccp4-9/bin/ccp4.setup-sh"


@functools.lru_cache(maxsize=1)
def ccp4_env() -> dict[str, str]:
    """Return ``os.environ`` augmented with the variables CCP4's setup script
    exports. The script is sourced once in a clean shell and the resulting
    environment cached. Falls back to a plain copy of ``os.environ`` if the
    setup script is missing or fails, so callers can always pass ``env=`` safely.
    """
    env = dict(os.environ)
    if not os.path.exists(CCP4_SETUP):
        return env
    try:
        proc = subprocess.run(
            ["sh", "-c", f". '{CCP4_SETUP}' >/dev/null 2>&1 && env -0"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return env
    if proc.returncode != 0:
        return env
    for entry in proc.stdout.split("\0"):
        key, sep, value = entry.partition("=")
        if sep and key:
            env[key] = value
    return env

# ── constants (mirrored from mmdb-refactor-ui/backend/config.py) ─────────────
CXX            = "clang++"
AUTOBUILD_LIB  = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/lib"
COOT_BUILD_DIR = "/lmb/home/jdialpuri/Development/coot-dev/build-linux"
COOT_API_DIR   = AUTOBUILD_LIB  # kept for test/compile.py import
MMDB_API_NAME  = "mmdb2"        # kept for test/compile.py import
GEMMI_INCLUDE  = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
CLIPPER_INCLUDE = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
BOOST_INCLUDE  = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
MMDB_INCLUDE   = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
GSL_INCLUDE    = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
PNG_INCLUDE    = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
GLM_INCLUDE    = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
RDKIT_INCLUDE  = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include/rdkit"


def make_compile_cmd(oracle_cc: Path, output_bin: Path) -> str:
    includes = [PROJECT_ROOT, GEMMI_INCLUDE, RDKIT_INCLUDE]
    includes = " ".join(f'-I"{i}"' for i in includes)

    clipper_libraries = " ".join(f'-l{l}' for l in [
        "clipper-core", "clipper-ccp4", "clipper-cif", "clipper-cns",
        "clipper-contrib", "clipper-minimol", "clipper-mmdb", "clipper-phs",
    ])
    rdkit_libraries = " ".join(f'-lRDKit{l}' for l in [
        "GraphMol", "SmilesParse", "FileParsers", "RDGeneral",
        "RDStreams", "RDGeometryLib", "SubstructMatch", "Depictor",
        "MolTransforms", "RDInchiLib",
    ])
    # gsl_libraries = "-lgsl -lgslcblas"

    return (
        f'{CXX} -std=c++20 -fno-access-control "{oracle_cc}" -o "{output_bin}" '
        f'{includes} '
        f'-Wl,-rpath,{AUTOBUILD_LIB} '
        f'-Wl,-rpath,{COOT_BUILD_DIR} '
        f'-L "{COOT_BUILD_DIR}" -lcootapi '
        f'-L "{AUTOBUILD_LIB}" {clipper_libraries} {rdkit_libraries} -lmmdb2 -lstdc++'
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
        try:
            proc = subprocess.run(
                ["sh", str(script)],
                text=True,
                stdout=f,
                stderr=f,
                cwd=str(out_dir),
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            f.write("\n[compile_oracle] timed out after 180s\n")
            return False
    return proc.returncode == 0

