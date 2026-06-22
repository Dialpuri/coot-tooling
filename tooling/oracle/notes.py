"""Post-oracle knowledge extraction.

After a working oracle.cc exists, ask the model to fill a small fixed schema
capturing the empirical facts it just validated: how the receiver was
constructed, which input values actually worked, gotchas encountered, and
notes useful for a later gemmi port. Saved as `oracle/notes.json` so
downstream stages (test, gemmi) can inject the relevant fields into their
own prompts without re-discovering them.

Kept strictly structured — small models produce garbage when asked for
"notes" freeform; a schema forces either concrete content or empty fields.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..llm import chat as _backend_chat

NOTES_SCHEMA_KEYS = (
    "construction",
    "inputs_that_work",
    "gotchas",
    "caveats_for_gemmi_port",
    "behavior",
)

BEHAVIOR_SCHEMA_KEYS = (
    "purpose",
    "branches",
    "edge_cases",
    "output_meaning",
)

_SYSTEM_PROMPT = """\
You are extracting empirical facts from a working C++ oracle program.
Everything in the program has been validated (it compiles and runs), so you
are recording what IS TRUE, not what might be true.

Output ONLY a single JSON object matching this schema, wrapped in a ```json
code block. Every field is optional — leave arrays empty or omit the key if
you have nothing concrete to record. Do NOT speculate.

{
  "construction":            ["C++ snippets, in order, that build the receiver and its prerequisites"],
  "inputs_that_work":        {"<param name>": "<literal value used in the oracle>"},
  "gotchas":                 ["one-line warnings a later caller should know (null pointers, ordering, required init calls)"],
  "caveats_for_gemmi_port":  ["one-line notes about things that will be tricky when porting to gemmi (mutated state, MMDB-specific APIs with no gemmi equivalent, assumed iteration order, etc.)"]
}

Rules:
  * Each construction entry should be a standalone line or small block, as it
    would appear in real code (include semicolons).
  * Only record things demonstrated by the oracle. If the oracle never set a
    flag, don't invent a gotcha about it.
  * Keep every list entry under 120 characters.
  * If you genuinely have nothing for a field, use [] or {}.\
"""


_BEHAVIOR_SYSTEM_PROMPT = """\
You are documenting what a C++ function DOES, so a later engineer can
re-implement it against a different library (gemmi) without re-reading the
original line by line. You are given the original function source, a working
oracle program that calls it, and the oracle's actual INPUT/OUTPUT for one
run. The oracle output is GROUND TRUTH for the path it exercised; the rest of
the function you must read and interpret.

Output ONLY a single JSON object matching this schema, wrapped in a ```json
code block.

{
  "purpose":        "one or two plain-English sentences: what the function computes or returns",
  "branches":       ["each meaningful condition -> what it does. Append ' (unverified)' if the oracle run did not exercise that path"],
  "edge_cases":     ["special handling worth preserving in a port: empty inputs, nulls, boundary values, early returns"],
  "output_meaning": ["for each OUTPUT the oracle printed, what the value represents and how it is derived"]
}

Rules:
  * Interpret the logic — this is documentation, not a transcript. Do NOT
    paste the source back; explain it.
  * The oracle output proves ONE path. Any branch the inputs did not hit is
    inference: tag it ' (unverified)' so a reader does not over-trust it.
  * Keep every list entry under 160 characters. Be concrete, not vague.
  * Leave a field as [] or "" if you have nothing concrete for it.\
"""


def _chat(messages: list[dict], model: str) -> str:
    """Single completion via the selected backend (CT_BACKEND: ollama/openai).

    Routes through tooling.llm so notes extraction uses the same backend as the
    agents — a hand-rolled Ollama /api/chat call 404s against the vLLM/OpenAI
    server the agents actually talk to.
    """
    data = _backend_chat(messages, model, tools=[])
    return (data.get("message") or {}).get("content") or ""


def _extract_json(text: str) -> dict | None:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else text.strip()
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _coerce_schema(raw: dict) -> dict:
    """Keep only known keys; drop unknown fields; coerce types."""
    out: dict = {}
    if isinstance(raw.get("construction"), list):
        out["construction"] = [str(x) for x in raw["construction"] if x]
    if isinstance(raw.get("inputs_that_work"), dict):
        out["inputs_that_work"] = {
            str(k): str(v) for k, v in raw["inputs_that_work"].items()
        }
    if isinstance(raw.get("gotchas"), list):
        out["gotchas"] = [str(x) for x in raw["gotchas"] if x]
    if isinstance(raw.get("caveats_for_gemmi_port"), list):
        out["caveats_for_gemmi_port"] = [str(x) for x in raw["caveats_for_gemmi_port"] if x]
    if isinstance(raw.get("behavior"), dict):
        beh = _coerce_behavior(raw["behavior"])
        if beh:
            out["behavior"] = beh
    return out


def _coerce_behavior(raw: dict) -> dict:
    """Keep only known behavior keys; drop unknown fields; coerce types."""
    out: dict = {}
    if isinstance(raw.get("purpose"), str) and raw["purpose"].strip():
        out["purpose"] = raw["purpose"].strip()
    for key in ("branches", "edge_cases", "output_meaning"):
        if isinstance(raw.get(key), list):
            vals = [str(x) for x in raw[key] if str(x).strip()]
            if vals:
                out[key] = vals
    return out


def extract_oracle_notes(
    oracle_cc_text: str,
    function_qname: str,
    model: str,
) -> dict | None:
    """Call the model to fill the notes schema. Returns None on any failure."""
    user_content = (
        f"Function observed: `{function_qname}`\n\n"
        f"Working oracle.cc:\n```cpp\n{oracle_cc_text}\n```\n\n"
        "Fill the JSON schema."
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    try:
        content = _chat(messages, model)
    except Exception:
        return None
    raw = _extract_json(content)
    if raw is None:
        return None
    return _coerce_schema(raw)


def summarise_behavior(
    original_src: str,
    oracle_cc_text: str,
    oracle_output: str,
    function_qname: str,
    model: str,
) -> dict | None:
    """Document what the function does, grounded in a working oracle run.

    Separate from `extract_oracle_notes`: that pass records empirical facts
    with no speculation; this pass interprets the logic (purpose, branches,
    edge cases) using the original source plus the oracle's actual output as
    ground truth. Returns a behavior dict, or None on any failure.
    """
    user_content = (
        f"Function: `{function_qname}`\n\n"
        f"Original source:\n```cpp\n{original_src}\n```\n\n"
        f"Working oracle.cc that calls it:\n```cpp\n{oracle_cc_text}\n```\n\n"
        f"Oracle INPUT/OUTPUT from the run (ground truth):\n```\n{oracle_output}\n```\n\n"
        "Fill the JSON schema."
    )
    messages = [
        {"role": "system", "content": _BEHAVIOR_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    try:
        content = _chat(messages, model)
    except Exception:
        return None
    raw = _extract_json(content)
    if raw is None:
        return None
    return _coerce_behavior(raw)


def save_notes(notes: dict, path: Path) -> None:
    path.write_text(json.dumps(notes, indent=2) + "\n")


def load_notes(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return _coerce_schema(raw) if isinstance(raw, dict) else None


def render_notes_for_prompt(notes: dict, audience: str) -> str:
    """Render a notes dict into a compact prompt fragment.

    audience: "test" or "gemmi" — controls which fields are included.
    """
    lines: list[str] = []
    beh = notes.get("behavior")
    if beh:
        if beh.get("purpose"):
            lines.append(f"What this function does: {beh['purpose']}")
        if beh.get("branches"):
            lines.append("Branches / conditions (unverified = oracle did not exercise):")
            lines.extend(f"  - {b}" for b in beh["branches"])
        if beh.get("edge_cases"):
            lines.append("Edge cases to preserve:")
            lines.extend(f"  - {e}" for e in beh["edge_cases"])
        if beh.get("output_meaning"):
            lines.append("What the outputs mean:")
            lines.extend(f"  - {o}" for o in beh["output_meaning"])
    if notes.get("construction"):
        lines.append("Proven construction (use this verbatim — already validated):")
        lines.extend(f"  {c}" for c in notes["construction"])
    if notes.get("inputs_that_work"):
        lines.append("Inputs known to work:")
        for k, v in notes["inputs_that_work"].items():
            lines.append(f"  {k} = {v}")
    if notes.get("gotchas"):
        lines.append("Gotchas from the oracle run:")
        lines.extend(f"  - {g}" for g in notes["gotchas"])
    if audience == "gemmi" and notes.get("caveats_for_gemmi_port"):
        lines.append("Port caveats (MMDB → gemmi):")
        lines.extend(f"  - {c}" for c in notes["caveats_for_gemmi_port"])
    return "\n".join(lines)
