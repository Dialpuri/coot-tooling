"""Soft, data-driven advisories appended to a FAILING run's output.

Each advisory lives in ``advisories/<name>.md`` with a tiny frontmatter block::

    ---
    name: protein-geometry-init
    when: (?<![\\w&*])protein_geometry\\s+...
    unless: init_standard
    ---
    ADVISORY (possible cause): ...

``when`` (required) is a regex that must match the code under inspection;
``unless`` (optional) is a regex that must NOT match. When both conditions
hold, the body text is surfaced.

Advisories are NEVER blocking — they are appended to an already-failing
oracle/test/gemmi run to point at a likely fix, so a false positive only costs
a few wasted tokens, never a rejected-but-valid port. To add one, drop a new
``advisories/*.md`` file; the loader auto-wires it (mirrors how ``overrides/``
auto-wires construction recipes in ``render.py``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ADVISORIES_DIR = Path(__file__).parent / "advisories"


@dataclass(frozen=True)
class Advisory:
    name: str
    when: re.Pattern
    unless: re.Pattern | None
    message: str

    def matches(self, code: str) -> bool:
        if not self.when.search(code):
            return False
        if self.unless is not None and self.unless.search(code):
            return False
        return True


def _parse_advisory(path: Path) -> Advisory | None:
    """Parse one advisory file. Returns None (and is silently skipped) if the
    frontmatter is malformed or the `when` regex doesn't compile — a broken
    advisory file must never crash a run."""
    text = path.read_text()
    if not text.lstrip().startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    _, frontmatter, body = parts
    meta: dict[str, str] = {}
    for line in frontmatter.strip().splitlines():
        key, sep, val = line.partition(":")
        if sep:
            meta[key.strip()] = val.strip()
    when = meta.get("when")
    if not when:
        return None
    try:
        when_rx = re.compile(when)
        unless_rx = re.compile(meta["unless"]) if meta.get("unless") else None
    except re.error:
        return None
    message = body.strip()
    if not message:
        return None
    return Advisory(
        name=meta.get("name", path.stem),
        when=when_rx,
        unless=unless_rx,
        message=message,
    )


def load_advisories() -> list[Advisory]:
    """Load every valid advisory in ADVISORIES_DIR, sorted by filename."""
    if not ADVISORIES_DIR.is_dir():
        return []
    out: list[Advisory] = []
    for p in sorted(ADVISORIES_DIR.glob("*.md")):
        adv = _parse_advisory(p)
        if adv is not None:
            out.append(adv)
    return out


def code_advisories(code: str) -> list[str]:
    """Return the message of every advisory whose trigger matches `code`.

    Pass the source text of the failing artefact (oracle.cc / test.cc /
    function.hh+cc). Empty input yields no advisories.
    """
    if not code:
        return []
    return [a.message for a in load_advisories() if a.matches(code)]
