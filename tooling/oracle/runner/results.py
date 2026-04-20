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

    @property
    def ran(self) -> bool:
        return self.returncode != -1

    def summary(self) -> str:
        lines = ["OK" if self.success else f"FAILED (exit {self.returncode})"]
        for name, value in self.inputs.items():
            lines.append(f"  INPUT  {name}: {value}")
        for name, value in self.outputs.items():
            lines.append(f"  OUTPUT {name}: {value}")
        if not self.success and self.stderr.strip():
            lines.append(f"  stderr: {self.stderr.strip()[:200]}")
        return "\n".join(lines)


_INPUT_RE  = re.compile(r"^INPUT\s+(.+?):\s*(.*)$",  re.MULTILINE)
_OUTPUT_RE = re.compile(r"^OUTPUT\s+(.+?):\s*(.*)$", re.MULTILINE)


def parse_output(returncode: int, stdout: str, stderr: str) -> OracleResult:
    inputs  = dict(_INPUT_RE.findall(stdout))
    outputs = dict(_OUTPUT_RE.findall(stdout))

    return OracleResult(
        success=returncode == 0,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        inputs=inputs,
        outputs=outputs,
    )


def save_result(path: Path, result: OracleResult) -> None:
    path.write_text(json.dumps(asdict(result), indent=2))


def load_result(path: Path) -> OracleResult:
    data = json.loads(path.read_text())
    return OracleResult(**data)
