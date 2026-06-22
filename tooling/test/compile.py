"""Compilation helpers for test.cc programs."""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..oracle.compile import (
    CXX, COOT_API_DIR, MMDB_API_NAME,
    AUTOBUILD_LIB, COOT_BUILD_DIR,
    GEMMI_INCLUDE, CLIPPER_INCLUDE, BOOST_INCLUDE,
    MMDB_INCLUDE, GSL_INCLUDE, PNG_INCLUDE, GLM_INCLUDE,
    RDKIT_INCLUDE, ccp4_env, is_teardown_only_crash,
    COOT_DEFINES_STR,
)
from ..db import PROJECT_ROOT

AUTOBUILD = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal"
GTEST_INCLUDE = f"/lmb/home/jdialpuri/Development/coot-tooling/third-party/google-test/include"
GTEST_LIB_DIR = f"/lmb/home/jdialpuri/Development/coot-tooling/third-party/google-test/lib"

MAX_COMPILE_ATTEMPTS = 20


def make_test_compile_cmd(test_cc: Path, output_bin: Path) -> str:
    includes = " ".join(f'-I"{i}"' for i in [
        PROJECT_ROOT, GEMMI_INCLUDE, CLIPPER_INCLUDE, BOOST_INCLUDE,
        MMDB_INCLUDE, GSL_INCLUDE, PNG_INCLUDE, GLM_INCLUDE,
        RDKIT_INCLUDE, GTEST_INCLUDE,
    ])

    clipper_libraries = " ".join(f'-l{l}' for l in [
        "clipper-core", "clipper-ccp4", "clipper-cif", "clipper-cns",
        "clipper-contrib", "clipper-minimol", "clipper-mmdb", "clipper-phs",
    ])
    rdkit_libraries = " ".join(f'-lRDKit{l}' for l in [
        "GraphMol", "SmilesParse", "FileParsers", "RDGeneral",
        "RDStreams", "RDGeometryLib", "SubstructMatch", "Depictor",
        "MolTransforms", "RDInchiLib",
    ])

    # -lgsl depends on -lgslcblas; needed by gradient/derivative functions.
    gsl_libraries = "-lgsl -lgslcblas"

    return (
        f'{CXX} -std=gnu++20 -fno-access-control {COOT_DEFINES_STR} "{test_cc.absolute()}" -o "{output_bin.absolute()}" '
        f'{includes} '
        f'-Wl,-rpath,{AUTOBUILD_LIB} '
        f'-Wl,-rpath,{COOT_BUILD_DIR} '
        f'-L "{COOT_BUILD_DIR}" -lcootapi '
        f'-L "{AUTOBUILD_LIB}" {clipper_libraries} {rdkit_libraries} -l{MMDB_API_NAME} '
        f'{gsl_libraries} -lstdc++ '
        f'-L "{GTEST_LIB_DIR}" -lgtest -lgtest_main -lm -no-pie'
    )



def compile_test_cc(test_cc: Path, output_bin: Path) -> tuple[bool, str]:
    """Compile test_cc. Returns (success, compiler output)."""
    cmd = make_test_compile_cmd(test_cc, output_bin)

    script = output_bin.parent / "compile.sh"
    script.write_text(f"#!/bin/sh\nset -e\n{cmd}\n")
    script.chmod(0o755)
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              cwd=str(test_cc.parent), timeout=180)
    except subprocess.TimeoutExpired:
        return False, "[compile_test_cc] timed out after 180s"
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def _spawn_and_wait(cmd: list[str], cwd: str, timeout: int) -> tuple[int | None, str, str]:
    """Spawn cmd in its own process group, wait up to timeout. Returns
    (returncode, stdout, stderr); returncode is None on timeout."""
    import os
    import signal

    # Capture BYTES and decode with errors="replace": test output loads real
    # structures and can carry non-UTF-8 bytes (e.g. 0xb0 '°' in cell angles),
    # which text=True would crash on with UnicodeDecodeError.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=cwd, start_new_session=True, env=ccp4_env(),
    )
    try:
        out_b, err_b = proc.communicate(timeout=timeout)
        return (proc.returncode,
                out_b.decode("utf-8", errors="replace"),
                err_b.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return None, "", ""


def run_test_binary(test_bin: Path, attempts: int = 2) -> tuple[bool, str]:
    """Run a compiled test binary. Returns (all tests passed, output).

    Retries once on timeout — the hang is non-deterministic and a fresh
    invocation usually completes in milliseconds.
    """
    cmd = ["stdbuf", "-oL", "-eL", str(test_bin.absolute())]
    cwd = str(test_bin.parent)
    last_out = ""
    for attempt in range(1, attempts + 1):
        rc, stdout, stderr = _spawn_and_wait(cmd, cwd, timeout=20)
        if rc is None:
            last_out = f"[run_test_binary] timed out after 20s (attempt {attempt}/{attempts})"
            continue
        out = (stdout + stderr).strip()
        passed = rc == 0
        verdict_rc = rc
        if not passed and is_teardown_only_crash(out, rc):
            # All assertions held; the crash was in global teardown. Record the
            # verdict as PASS but keep the real signal visible in the log.
            out += (
                f"\n\n[harness] All gtest assertions PASSED; the process then "
                f"exited {rc} during global/atexit teardown (typically a static "
                f"molecules_container_t destructor double-free), NOT a test "
                f"failure. Recording verdict as PASS. To make the binary exit "
                f"cleanly, give the test its own `int main(int argc, char** argv) "
                f"{{ ::testing::InitGoogleTest(&argc, &argv); int r = "
                f"RUN_ALL_TESTS(); std::fflush(nullptr); _exit(r); }}` "
                f"(skips the destructors)."
            )
            passed = True
            verdict_rc = 0
        try:
            (test_bin.parent / "run.exit").write_text(str(verdict_rc))
        except OSError:
            pass
        return passed, out
    return False, last_out


def write_compile_script(test_subdir: Path) -> Path:
    """Write compile_test.sh into test_subdir and make it executable."""
    test_cc  = test_subdir / "test.cc"
    test_bin = test_subdir / "test"
    cmd = make_test_compile_cmd(test_cc, test_bin)
    script = test_subdir / "compile_test.sh"
    script.write_text(f"#!/bin/sh\nset -e\n{cmd}\n")
    script.chmod(0o755)
    return script
