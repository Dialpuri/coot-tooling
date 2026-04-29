"""Structured oracle result parsing and persistence."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class OracleResult:
    success:    bool
    returncode: int
    stdout:     str
    stderr:     str
    inputs:     dict[str, str] = field(default_factory=dict)
    outputs:    dict[str, str] = field(default_factory=dict)
    cases:      list[dict]     = field(default_factory=list)

    @property
    def ran(self) -> bool:
        return self.returncode != -1

    def summary(self) -> str:
        lines = ["OK" if self.success else f"FAILED (exit {self.returncode})"]
        if self.cases:
            for i, case in enumerate(self.cases, 1):
                lines.append(f"  Case {i}:")
                for name, value in case["inputs"].items():
                    lines.append(f"    INPUT  {name}: {value}")
                for name, value in case["outputs"].items():
                    lines.append(f"    OUTPUT {name}: {value}")
        else:
            for name, value in self.inputs.items():
                lines.append(f"  INPUT  {name}: {value}")
            for name, value in self.outputs.items():
                lines.append(f"  OUTPUT {name}: {value}")
        if not self.success and self.stderr.strip():
            lines.append(f"  stderr: {self.stderr.strip()[:200]}")
        return "\n".join(lines)


_INPUT_RE  = re.compile(r"^INPUT\s+(.+?):\s*(.*)$",  re.MULTILINE)
_OUTPUT_RE = re.compile(r"^OUTPUT\s+(.+?):\s*(.*)$", re.MULTILINE)


def _parse_cases(stdout: str) -> list[dict]:
    """Group INPUT/OUTPUT lines into discrete test cases.

    A new case begins when an INPUT line appears after at least one OUTPUT.
    Within a case, if the same key appears multiple times (e.g. an oracle that
    prints all atom inputs then all atom outputs), the inputs and outputs are
    paired positionally and split into individual cases — one per repeated key.
    """
    cases: list[dict] = []
    cur_inputs:  list[tuple[str, str]] = []
    cur_outputs: list[tuple[str, str]] = []
    seen_output = False

    def _flush() -> None:
        nonlocal cur_inputs, cur_outputs, seen_output
        if not cur_inputs and not cur_outputs:
            return
        in_keys  = [k for k, _ in cur_inputs]
        out_keys = [k for k, _ in cur_outputs]
        has_collision = len(in_keys) != len(set(in_keys)) or \
                        len(out_keys) != len(set(out_keys))
        if has_collision and len(cur_inputs) == len(cur_outputs):
            # Pair by position — e.g. N INPUT residue lines then N OUTPUT residue lines
            for (ik, iv), (ok, ov) in zip(cur_inputs, cur_outputs):
                cases.append({"inputs": {ik: iv}, "outputs": {ok: ov}})
        elif has_collision:
            # Counts differ — index the keys to preserve all values
            cases.append({
                "inputs":  {f"{k}[{i}]": v for i, (k, v) in enumerate(cur_inputs)},
                "outputs": {f"{k}[{i}]": v for i, (k, v) in enumerate(cur_outputs)},
            })
        else:
            cases.append({"inputs": dict(cur_inputs), "outputs": dict(cur_outputs)})
        cur_inputs.clear()
        cur_outputs.clear()
        seen_output = False

    for line in stdout.splitlines():
        inp = re.match(r"^INPUT\s+(.+?):\s*(.*)$", line)
        out = re.match(r"^OUTPUT\s+(.+?):\s*(.*)$", line)
        if inp:
            if seen_output:
                _flush()
            cur_inputs.append((inp.group(1).strip(), inp.group(2).strip()))
        elif out:
            cur_outputs.append((out.group(1).strip(), out.group(2).strip()))
            seen_output = True

    _flush()
    return cases


def parse_output(returncode: int, stdout: str, stderr: str) -> OracleResult:
    inputs  = dict(_INPUT_RE.findall(stdout))
    outputs = dict(_OUTPUT_RE.findall(stdout))
    cases   = _parse_cases(stdout)
    return OracleResult(
        success=returncode == 0,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        inputs=inputs,
        outputs=outputs,
        cases=cases,
    )


def save_result(path: Path, result: OracleResult) -> None:
    path.write_text(json.dumps(asdict(result), indent=2))


def load_result(path: Path) -> OracleResult:
    data = json.loads(path.read_text())
    data.setdefault("cases", [])
    return OracleResult(**data)
