#!/usr/bin/env python3
"""
Layer 2: Embed every function definition and build a FAISS index.

Reads:  ast-data/code_graph.db  (produced by extract_graph.py)
Writes: ast-data/index.faiss    (FAISS IndexFlatIP, cosine similarity)
        ast-data/index_meta.json (maps FAISS row -> function record)

Run with the faiss conda env active:
  conda activate faiss
  python ast-script/build_index.py
"""
import json
import os
import sqlite3
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH    = Path(__file__).parent.parent / "ast-data" / "code_graph.db"
INDEX_PATH = Path(__file__).parent.parent / "ast-data" / "index.faiss"
META_PATH  = Path(__file__).parent.parent / "ast-data" / "index_meta.json"
PROJECT_ROOT = "/Users/dialpuri/lmb/coot"

# MODEL_NAME  = "BAAI/bge-small-en-v1.5"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE  = 64
MAX_CODE_CHARS = 3000   # truncate very long function bodies before embedding


# ---------------------------------------------------------------------------
# Context document assembly
# ---------------------------------------------------------------------------

def build_document(fn: dict, type_summaries: list[str], callees: list[str]) -> str:
    """Produce a single string capturing everything the model needs to understand this function."""
    rel_file = fn["file"].replace(PROJECT_ROOT + "/", "")

    parts = [
        f"Function: {fn['qualified_name']}",
        f"File: {rel_file}:{fn['line_start']}",
    ]

    if callees:
        parts.append("Calls: " + ", ".join(callees[:20]))

    if type_summaries:
        parts.append("\nUsed types:")
        for s in type_summaries:
            parts.append(s)

    if fn["source_code"]:
        code = fn["source_code"]
        if len(code) > MAX_CODE_CHARS:
            code = code[:MAX_CODE_CHARS] + "\n// ... (truncated)"
        parts.append("\nSource:")
        parts.append(code)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Load all function definitions (declarations have no source, less useful)
    functions = conn.execute("""
        SELECT f.id, f.qualified_name, f.display_name,
               fi.path AS file, f.line_start, f.line_end,
               f.kind, f.source_code
        FROM functions f
        JOIN files fi ON fi.id = f.file_id
        WHERE f.is_definition = 1
    """).fetchall()

    print(f"Functions to embed: {len(functions)}")

    # Pre-load callee names per function  (caller_id -> [callee_name, ...])
    callee_map: dict[int, list[str]] = {}
    for row in conn.execute("SELECT caller_id, callee_qualified_name FROM calls"):
        callee_map.setdefault(row[0], []).append(row[1])

    # Pre-load type summaries per function (function_id -> [summary, ...])
    type_map: dict[int, list[str]] = {}
    for row in conn.execute("""
        SELECT u.function_id, t.summary
        FROM uses_type u
        JOIN types t ON t.qualified_name = u.type_qualified_name
        WHERE t.summary IS NOT NULL
    """):
        type_map.setdefault(row[0], []).append(row[1])

    conn.close()

    # Build one document per function
    documents = []
    metadata  = []
    for fn in functions:
        fn_id    = fn["id"]
        doc      = build_document(
            dict(fn),
            type_map.get(fn_id, []),
            callee_map.get(fn_id, []),
        )
        documents.append(doc)
        metadata.append({
            "faiss_id":      len(metadata),
            "function_id":   fn_id,
            "qualified_name": fn["qualified_name"],
            "display_name":  fn["display_name"],
            "file":          fn["file"],
            "line_start":    fn["line_start"],
            "line_end":      fn["line_end"],
        })

    # Embed
    print(f"Loading model: {MODEL_NAME}  (downloads on first run)")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {len(documents)} documents in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        documents,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors -> cosine sim via dot product
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)

    # Build FAISS index (IndexFlatIP = exact inner-product = cosine on unit vecs)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(metadata, indent=2))

    print(
        f"\nIndex written to  {INDEX_PATH}  ({index.ntotal} vectors, dim={dim})\n"
        f"Metadata written to {META_PATH}"
    )


if __name__ == "__main__":
    main()
