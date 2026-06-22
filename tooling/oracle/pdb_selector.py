"""The multi-fixture verification panel and its prompt overview.

The oracle is run across a FIXED panel of structurally-distinct structures
(protein / protein+H / ligand-only / protein-ligand / RNA / glycoprotein) — a
correct port yields a different result on each, so a value hardcoded to one
structure fails the others. There is no per-function structure selection any
more: the oracle reads its path from argv and navigates generically, the runner
runs the whole panel, and the agent inspects whichever fixtures it needs via the
`inspect_pdb` tool. `panel_overview()` shows the agent the shape of every fixture
up front so it can write navigation that works on all of them.
"""
from __future__ import annotations

from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "test-data"

# The structure baked into a generated oracle as its `argc < 2` fallback so the
# binary is runnable by hand. The harness always passes argv[1], so this is only
# a convenience default — NOT a per-function choice.
DEFAULT_FIXTURE = "example.pdb"

# One-line description per fixture, shown in the panel overview.
PDB_CATALOG: dict[str, str] = {
    "example.pdb":
        "standard protein (transferase 2VTQ, chains A+B, no hydrogens)",
    "example-hydrogen.pdb":
        "protein with explicit hydrogen atoms added",
    "example-ligand.cif":
        "small-molecule ligand only (no protein) — restraint/rdkit/SMILES dict",
    "example-protein-ligand.cif":
        "protein-ligand complex (protein + bound small molecule)",
    "example-na.pdb":
        "nucleic acid (RNA) — chains of A/G/C/U residues, Mg ions, waters",
    "example-glycosylated.cif":
        "glycoprotein — protein with N-linked glycans (NAG/MAN/BMA/GAL)",
}

# Optional structural notes (e.g. where the ligand of interest is) folded into
# the panel overview so the agent knows what to navigate to.
_PDB_NOTES: dict[str, str] = {
    "example-protein-ligand.cif":
        "ligand LZA at residue 1299, chain A is the bound small molecule",
    "example-ligand.cif":
        "the ligand is LZA",
}

# ── the panel ─────────────────────────────────────────────────────────────────
# Order = primary first (drives OraclePanelResult.primary()).
FIXTURE_PANEL: list[str] = [
    "example.pdb",                 # standard protein
    "example-hydrogen.pdb",        # protein with explicit hydrogens
    "example-ligand.cif",          # small-molecule ligand only (no protein)
    "example-protein-ligand.cif",  # protein + bound small-molecule complex
    "example-na.pdb",              # nucleic acid (RNA)
    "example-glycosylated.cif",    # glycoprotein (protein + N-linked glycans)
]


def pdb_path(filename: str) -> Path:
    return TEST_DATA_DIR / filename


def paired_mtz(structure_file: str) -> str | None:
    """Absolute path of the map coefficients paired with a structure, or None.

    Convention: `<stem>.mtz` next to `<stem>.pdb/.cif` (e.g. example-na.mtz).
    """
    stem = structure_file.rsplit(".", 1)[0]
    cand = TEST_DATA_DIR / f"{stem}.mtz"
    return str(cand) if cand.exists() else None


def fixture_panel() -> list[tuple[str, str | None]]:
    """(structure_path, mtz_path|None) for each panel fixture present on disk."""
    out: list[tuple[str, str | None]] = []
    for f in FIXTURE_PANEL:
        p = TEST_DATA_DIR / f
        if p.exists():
            out.append((str(p), paired_mtz(f)))
    return out


def _structure_shape(path: Path) -> str:
    """Compact chain/residue-kind shape of a structure, or '' if unreadable.

    Used in the panel overview so the agent sees the variety it must handle.
    Robust across .pdb and coordinate .cif via gemmi; a non-coordinate file
    (e.g. a restraint-only .cif) yields '' and falls back to the description.
    """
    try:
        import gemmi
        st = gemmi.read_structure(str(path))
    except Exception:
        return ""
    if not st or not len(st):
        return ""
    bits: list[str] = []
    for chain in st[0]:
        names: list[str] = []
        for res in chain:
            if res.name not in names:
                names.append(res.name)
        shown = ", ".join(names[:6]) + (", …" if len(names) > 6 else "")
        bits.append(f"{chain.name}({len(chain)} res: {shown})")
        if len(bits) >= 4:
            bits.append("…")
            break
    return "chains " + "; ".join(bits)


def panel_overview() -> str:
    """The FIXTURE PANEL block injected into the oracle prompt.

    One entry per panel fixture: its path, description, a compact structural
    shape, and any navigation note. Replaces the old single-structure
    STRUCTURE CONTENT block so the agent writes structure-agnostic navigation.
    """
    lines = [
        "FIXTURE PANEL — the oracle is run against EACH of these; write "
        "navigation that works on all of them (use inspect_pdb(fixture=...) for "
        "detail). Read your structure from argv[1]:",
    ]
    for fname in FIXTURE_PANEL:
        p = TEST_DATA_DIR / fname
        if not p.exists():
            continue
        desc = PDB_CATALOG.get(fname, "")
        lines.append(f"  - {p}")
        lines.append(f"      {desc}")
        shape = _structure_shape(p)
        if shape:
            lines.append(f"      {shape}")
        note = _PDB_NOTES.get(fname)
        if note:
            lines.append(f"      note: {note}")
    return "\n".join(lines)
