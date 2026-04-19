#!/usr/bin/env python3
"""
Generate one-sentence summaries for functions that have no comment,
using a local Ollama model.

Reads/writes: ast-data/code_graph.db  (updates functions.comment)

Usage:
  ollama serve          # if not already running
  ollama pull <model>   # first time only
  python ast-script/summarise_functions.py
  python ast-script/summarise_functions.py --model qwen2.5-coder:1.5b --batch 50
"""
import argparse
import sqlite3
import time
import urllib.request
import urllib.error
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH      = Path(__file__).parent.parent / "ast-data" / "code_graph.db"
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL        = "gemma4:e4b"   # e.g. "qwen2.5-coder:1.5b"
BATCH_SIZE   = 20     # commit to DB every N summaries
MAX_SRC_CHARS = 2000  # truncate very long functions before sending

PROMPT_TEMPLATE = """\
Summarize the following C++ function in one sentence (maximum 20 words).
Focus on what it does, not how it does it.
Reply with only the summary sentence — no explanation, no punctuation other than a full stop. 
Don't refer to the function as a thing by using it, just say what it does.

{source}
"""


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def ollama_generate(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "think":  False,   # disable reasoning tokens (gemma4 and other thinking models)
        "options": {"temperature": 0.1, "num_predict": 80},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip()


def check_ollama(model: str) -> None:
    """Abort early with a clear message if Ollama isn't reachable."""
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=3)
    except Exception:
        raise SystemExit(
            "Ollama is not running.  Start it with:  ollama serve"
        )
    if model == "REPLACE_WITH_MODEL_NAME":
        raise SystemExit(
            "Set MODEL in this script (or pass --model) to a model you have pulled.\n"
            "Available models:  ollama list"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _process_row(row: sqlite3.Row, model: str) -> tuple[int, str, str]:
    """Called in a worker thread. Returns (id, qualified_name, summary)."""
    src = row["source_code"]
    if len(src) > MAX_SRC_CHARS:
        src = src[:MAX_SRC_CHARS] + "\n// ..."
    prompt  = PROMPT_TEMPLATE.format(source=src)
    summary = ollama_generate(prompt, model)
    return row["id"], row["qualified_name"], summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise undocumented functions via Ollama")
    parser.add_argument("--model",   default=MODEL,      help="Ollama model name")
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE, help="DB commit interval")
    parser.add_argument("--limit",   type=int, default=0,  help="Stop after N summaries (0 = all)")
    parser.add_argument("--workers", type=int, default=4,  help="Parallel Ollama requests (default 4)")
    args = parser.parse_args()

    check_ollama(args.model)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT id, qualified_name, display_name, source_code
        FROM functions
        WHERE is_definition = 1
          AND source_code IS NOT NULL
          AND source_code != ''
          AND (comment IS NULL OR comment = '')
        ORDER BY id
    """).fetchall()

    total = len(rows) if not args.limit else min(len(rows), args.limit)
    print(f"Functions without comments: {len(rows)}  (processing: {total}, workers: {args.workers})")

    updated = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_row, row, args.model): row
            for row in rows[:total]
        }
        for i, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                fn_id, qname, summary = future.result()
            except urllib.error.URLError as e:
                print(f"  [{i}/{total}] NETWORK ERROR {row['qualified_name'][:60]}: {e}")
                continue
            except Exception as e:
                print(f"  [{i}/{total}] ERROR {row['qualified_name'][:60]}: {e}")
                continue

            conn.execute("UPDATE functions SET comment = ? WHERE id = ?", (summary, fn_id))
            updated += 1
            print(f"  [{i}/{total}] {qname[:70]}")
            print(f"           → {summary}")

            if updated % args.batch == 0:
                conn.commit()

    conn.commit()
    conn.close()
    print(f"\nDone. {updated} functions summarised.")


if __name__ == "__main__":
    main()
