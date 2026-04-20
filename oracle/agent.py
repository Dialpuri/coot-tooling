"""
Agentic oracle generation — gemma4 calls tools to explore the codebase on
demand rather than receiving a pre-built context dump.

The model is given the function source and a lean prompt, then iteratively
calls tools (read_file, lookup_function, lookup_type, list_methods,
get_callers, search_functions) until it has enough context to write oracle.cc.

A human-readable trace of every tool call and result is saved alongside the
oracle so you can audit what the model looked up.
"""
from __future__ import annotations

import json
import re
import sqlite3
import textwrap
import urllib.request
import urllib.error
from pathlib import Path

from .db import (
    PROJECT_ROOT,
    get_function,
    get_type,
    get_type_methods,
    get_callers_with_source,
    get_class_functions,
)
from .render import INCLUDE_ROOTS, _to_include, _load_override, MMDB_MANAGER_SNIPPET

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

NOTES_DIR     = Path(__file__).parent / "notes"
ANSWER_MARKER = "## Answer"
TEST_DATA_DIR = Path(__file__).parent.parent / "test-data"

# Paths the model is allowed to read.
ALLOWED_READ_ROOTS = [PROJECT_ROOT] + INCLUDE_ROOTS

AGENT_SYSTEM_PROMPT = f"""\
You are writing a complete, compilable C++ program (oracle.cc) that observes
the inputs and outputs of the function given by the user.

Requirements for oracle.cc:
  1. Be self-contained — hardcode the test file paths below, do not use argc/argv.
       PDB: {TEST_DATA_DIR}/example.pdb
       MTZ: {TEST_DATA_DIR}/example.mtz
  2. Load the structure using the hardcoded path.
  3. Navigate the structure to reach a valid receiver/input for the function.
  4. Call the function.
  5. Print every input value and every meaningful output value:
       INPUT  <name>: <value>
       OUTPUT <name>: <value>
  6. Use this pattern to produce a few edge cases too. 

Use the tools to look up types, headers, callers, and construction patterns
until you have enough context. Use C++ code where possible, avoid C style code.
Then output the complete program in a ```cpp block.\
"""

# ── tool schema ───────────────────────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a C++ source or header file. Use to inspect includes, "
                "type definitions, or method bodies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_function",
            "description": (
                "Return source code and documentation for a function by its "
                "fully-qualified name, e.g. 'coot::molecule_t::cid_to_residue'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "qualified_name": {"type": "string"},
                },
                "required": ["qualified_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_type",
            "description": (
                "Return the class/struct definition and method list for a type. "
                "Accepts short names like 'Residue' or qualified names like "
                "'mmdb::Residue'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_methods",
            "description": "List all method signatures in a C++ class.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {"type": "string"},
                },
                "required": ["class_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_callers",
            "description": (
                "Return example functions that call the given function, showing "
                "real usage patterns and construction of receiver objects."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "qualified_name": {"type": "string"},
                },
                "required": ["qualified_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_header",
            "description": (
                "Given a type or function name, return its absolute file path "
                "and the #include directive to use. Call this before read_file "
                "when you need to inspect a header but don't know its path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Type or function qualified name"},
                },
                "required": ["name"],
            },
        },
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "leave_note",
    #         "description": (
    #             "When you are uncertain about something domain-specific and have to guess "
    #             "(e.g. what a valid CID string looks like, which chain to use, expected "
    #             "value ranges), call this to record your question. The user will be notified "
    #             "and can write an answer that will be available in future runs."
    #         ),
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "topic": {
    #                     "type": "string",
    #                     "description": "Short slug for the note, e.g. 'cid_format' or 'chain_id'",
    #                 },
    #                 "question": {
    #                     "type": "string",
    #                     "description": "What you are uncertain about and what would help you.",
    #                 },
    #             },
    #             "required": ["topic", "question"],
    #         },
    #     },
    # },
    {
        "type": "function",
        "function": {
            "name": "resolve_includes",
            "description": (
                "Check every #include \"...\" directive in your current draft. "
                "Reports which paths resolve correctly and, for those that do not, "
                "searches the coot source tree for a file with the same name and "
                "returns the correct #include path. Call this before finalising "
                "oracle.cc to catch wrong or missing headers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The current C++ draft (full file or just the #include lines)",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_functions",
            "description": (
                "Search for functions whose qualified name contains a substring. "
                "Useful when you don't know the exact name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name_fragment": {"type": "string"},
                },
                "required": ["name_fragment"],
            },
        },
    },
]


# ── tool implementations ──────────────────────────────────────────────────────

def _tool_read_file(path: str) -> str:
    real = Path(path).resolve()
    if not any(str(real).startswith(root) for root in ALLOWED_READ_ROOTS):
        return f"ERROR: path '{path}' is outside allowed roots."
    try:
        text = real.read_text(errors="replace")
    except OSError as e:
        return f"ERROR: {e}"
    # Cap at ~300 lines to avoid flooding the context.
    lines = text.splitlines()
    if len(lines) > 300:
        return "\n".join(lines[:300]) + f"\n\n... ({len(lines) - 300} more lines)"
    return text


def _tool_lookup_function(conn: sqlite3.Connection, qualified_name: str) -> str:
    row = get_function(conn, qualified_name)
    if not row:
        return f"Function '{qualified_name}' not found in DB."
    parts = []
    if row["comment"]:
        parts.append(f"// {row['comment']}")
    parts.append(row["source_code"] or "(no source)")
    return "\n".join(parts)


def _tool_lookup_type(conn: sqlite3.Connection, name: str) -> str:
    row = get_type(conn, name)
    if not row:
        return f"Type '{name}' not found in DB."
    methods = get_type_methods(conn, row["qualified_name"])
    lines = [f"// {row['kind']} {row['qualified_name']}  ({row['file']})"]
    lines.append(row["summary"] or "(no summary)")
    if methods:
        lines.append("\n// Methods:")
        for m in methods:
            comment = f"  // {m['comment']}" if m["comment"] else ""
            lines.append(f"  {m['display_name']}{comment}")
    return "\n".join(lines)


def _tool_list_methods(conn: sqlite3.Connection, class_name: str) -> str:
    qnames = get_class_functions(conn, class_name)
    if not qnames:
        return f"No methods found for '{class_name}'."
    return "\n".join(qnames)


def _tool_get_callers(conn: sqlite3.Connection, qualified_name: str) -> str:
    fn = get_function(conn, qualified_name)
    if not fn:
        return f"Function '{qualified_name}' not found."
    callers = get_callers_with_source(conn, fn["id"], limit=3)
    if not callers:
        return "No callers found."
    parts = []
    for c in callers:
        rel = c["file"].replace(PROJECT_ROOT + "/", "")
        parts.append(f"// {rel}")
        if c["comment"]:
            parts.append(f"// {c['comment']}")
        parts.append(c["source_code"].rstrip())
    return "\n\n".join(parts)


def _tool_find_header(conn: sqlite3.Connection, name: str) -> str:
    file_path = None
    row = get_type(conn, name)
    if row:
        file_path = row["file"]
    else:
        fn = get_function(conn, name)
        if fn:
            file_path = fn["file"]
    if not file_path:
        return f"No type or function '{name}' found in DB."
    include = _to_include(file_path)
    return f'Absolute path: {file_path}\n#include "{include}"'


def _tool_leave_note(topic: str, question: str) -> str:
    NOTES_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9_]", "_", topic.lower()).strip("_")
    path = NOTES_DIR / f"{slug}.md"
    if not path.exists():
        path.write_text(
            f"# {topic}\n\n"
            f"## Question\n{question}\n\n"
            f"{ANSWER_MARKER}\n<!-- Fill in your answer here -->\n"
        )
        return f"Note created: {path}"
    return f"Note already exists: {path} (not overwritten)"


def _load_notes() -> str | None:
    """Return all notes as a single context block, or None if the dir is empty."""
    if not NOTES_DIR.exists():
        return None
    notes = sorted(NOTES_DIR.glob("*.md"))
    if not notes:
        return None
    parts = []
    for p in notes:
        parts.append(f"=== {p.stem} ===\n{p.read_text().strip()}")
    return "\n\n".join(parts)


def _unanswered_notes() -> list[Path]:
    """Return notes that have no answer written yet."""
    if not NOTES_DIR.exists():
        return []
    unanswered = []
    for p in sorted(NOTES_DIR.glob("*.md")):
        text = p.read_text()
        marker_pos = text.find(ANSWER_MARKER)
        if marker_pos == -1:
            continue
        answer_body = text[marker_pos + len(ANSWER_MARKER):].strip()
        if not answer_body or answer_body.startswith("<!--"):
            unanswered.append(p)
    return unanswered


def _tool_resolve_includes(code: str) -> str:
    includes = re.findall(r'#include\s+"([^"]+)"', code)
    if not includes:
        return "No local #include \"...\" directives found in the supplied code."

    lines: list[str] = []
    for inc in includes:
        # Try resolving from each allowed root directly.
        found_at: list[str] = []
        for root in ALLOWED_READ_ROOTS:
            candidate = Path(root) / inc
            if candidate.exists():
                found_at.append(str(candidate))

        if found_at:
            lines.append(f'OK  #include "{inc}"  →  {found_at[0]}')
            continue

        # Not found at the given path — search by filename across the tree.
        filename = Path(inc).name
        matches: list[Path] = []
        for root in ALLOWED_READ_ROOTS:
            matches.extend(Path(root).rglob(filename))
        matches = sorted(set(matches))[:5]

        if not matches:
            lines.append(f'MISSING  #include "{inc}"  (no file named {filename!r} found in source tree)')
            continue

        lines.append(f'WRONG PATH  #include "{inc}"')
        for m in matches:
            # Express as a path relative to the first root that contains it.
            for root in ALLOWED_READ_ROOTS:
                try:
                    rel = m.relative_to(root)
                    lines.append(f'  use instead:  #include "{rel}"  (at {m})')
                    break
                except ValueError:
                    continue
            else:
                lines.append(f'  found at:  {m}  (no standard include root covers this)')

    return "\n".join(lines)


def _tool_search_functions(conn: sqlite3.Connection, name_fragment: str) -> str:
    rows = conn.execute("""
        SELECT DISTINCT qualified_name FROM functions
        WHERE qualified_name LIKE ?
        LIMIT 30
    """, (f"%{name_fragment}%",)).fetchall()
    if not rows:
        return f"No functions matching '{name_fragment}'."
    return "\n".join(r[0] for r in rows)


def _dispatch(conn: sqlite3.Connection, name: str, args: dict) -> str:
    if name == "read_file":
        return _tool_read_file(args["path"])
    if name == "lookup_function":
        return _tool_lookup_function(conn, args["qualified_name"])
    if name == "lookup_type":
        return _tool_lookup_type(conn, args["name"])
    if name == "list_methods":
        return _tool_list_methods(conn, args["class_name"])
    if name == "get_callers":
        return _tool_get_callers(conn, args["qualified_name"])
    if name == "find_header":
        return _tool_find_header(conn, args["name"])
    if name == "resolve_includes":
        return _tool_resolve_includes(args["code"])
    if name == "search_functions":
        return _tool_search_functions(conn, args["name_fragment"])
    if name == "leave_note":
        return _tool_leave_note(args["topic"], args["question"])
    return f"Unknown tool: {name}"


# ── Ollama chat API ───────────────────────────────────────────────────────────

def _chat(messages: list[dict], model: str, tools: list[dict]) -> dict:
    payload = json.dumps({
        "model":    model,
        "messages": messages,
        "tools":    tools,
        "stream":   False,
        "think":    True,
    }).encode()
    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


# ── agent loop ────────────────────────────────────────────────────────────────

def generate_with_agent(
    conn: sqlite3.Connection,
    function_qname: str,
    model: str,
    verbose: bool = False,
) -> tuple[str | None, str]:
    """Run the agentic oracle generation loop.

    Returns (oracle_code, trace_text).
    oracle_code is None if the function wasn't found or the model produced nothing.
    trace_text is a human-readable log of every tool call and result.
    """
    fn = get_function(conn, function_qname)
    if not fn:
        return None, "Function not found in DB."

    user_content = (
        f"FUNCTION TO OBSERVE: {function_qname}\n\n"
        f"File: {fn['file']}\n\n"
    )
    if fn["comment"]:
        user_content += f"// {fn['comment']}\n"
    user_content += fn["source_code"] or "(no source)"

    # Always include the MMDB Manager usage snippet — ReadCoorFile is an
    # external library not in the DB, so the agent can't find it via tools.
    user_content += "\n\n// === MMDB USAGE ===\n" + MMDB_MANAGER_SNIPPET

    # Load any curated notes (questions + answers) left by previous runs.
    notes_context = _load_notes()
    if notes_context:
        user_content += "\n\n// === DOMAIN NOTES ===\n" + notes_context

    # Attach the curated construction snippet for the containing class if one exists.
    if "::" in function_qname:
        class_qname = function_qname.rsplit("::", 1)[0]
        override = _load_override(class_qname)
        if override:
            user_content += (
                f"\n\n// === HOW TO CONSTRUCT {class_qname} ===\n"
                + override.rstrip()
            )

    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    trace_lines: list[str] = [
        f"=== AGENT TRACE: {function_qname} ===\n",
        f"[user]\n{textwrap.indent(user_content, '  ')}\n",
    ]

    max_turns = 12
    oracle_code: str | None = None

    for turn in range(max_turns):
        data = _chat(messages, model, TOOLS)
        msg  = data.get("message", {})
        tool_calls = msg.get("tool_calls") or []

        thinking          = msg.get("thinking", "") or ""
        assistant_content = msg.get("content",  "") or ""
        messages.append({"role": "assistant", "content": assistant_content,
                          "tool_calls": tool_calls})

        if thinking:
            if verbose:
                print(f"\n[thinking]\n{textwrap.indent(thinking, '  ')}\n")
            trace_lines.append(f"[thinking — turn {turn + 1}]\n{textwrap.indent(thinking, '  ')}\n")

        if not tool_calls:
            # Model is done — extract the oracle code from the response.
            trace_lines.append(f"[assistant — final]\n{textwrap.indent(assistant_content, '  ')}\n")
            import re
            m = re.search(r"```(?:cpp|c\+\+)?\n(.*?)```", assistant_content, re.DOTALL)
            oracle_code = m.group(1).strip() if m else assistant_content.strip()
            break

        # Execute each tool call and collect results.
        trace_lines.append(f"[assistant — turn {turn + 1}, {len(tool_calls)} tool call(s)]")
        tool_results: list[dict] = []

        for call in tool_calls:
            fn_info = call.get("function", {})
            name    = fn_info.get("name", "")
            args    = fn_info.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            if verbose:
                print(f"  tool: {name}({args})")

            result = _dispatch(conn, name, args)


            # Cap individual tool results to avoid blowing up the context.
            result_lines = result.splitlines()
            if len(result_lines) > 150:
                result = "\n".join(result_lines[:150]) + f"\n... ({len(result_lines) - 150} more lines)"

            trace_lines.append(f"  → {name}({json.dumps(args)})")
            trace_lines.append(textwrap.indent(result, "      ") + "\n")

            tool_results.append({
                "role":    "tool",
                "content": result,
            })

        messages.extend(tool_results)

    else:
        trace_lines.append("[agent] Max turns reached without a final answer.\n")

    # Notify about any notes that still need an answer.
    unanswered = _unanswered_notes()
    if unanswered:
        trace_lines.append("\n[notes] Unanswered questions — please fill in:")
        for p in unanswered:
            trace_lines.append(f"  {p}")
        if verbose:
            print("\n*** Unanswered notes — please fill in an answer: ***")
            for p in unanswered:
                print(f"  {p}")

    return oracle_code, "\n".join(trace_lines)
