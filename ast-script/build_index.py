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
import argparse
import json
import os
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_DB         = Path(__file__).parent.parent / "ast-data" / "code_graph.db"
DEFAULT_INDEX      = Path(__file__).parent.parent / "ast-data" / "index.faiss"
DEFAULT_META       = Path(__file__).parent.parent / "ast-data" / "index_meta.json"
DEFAULT_MODEL      = "google/embeddinggemma-300m"
DEFAULT_BATCH_SIZE = 32
MAX_CODE_CHARS     = 3000


# ---------------------------------------------------------------------------
# Context document assembly
# ---------------------------------------------------------------------------

def build_document(fn: dict, callees: list[str], project_root: str = "") -> str:
    """Produce a single string for embedding.

    Ordered so that the most library-agnostic signal (comment/summary) comes
    first — sentence transformers weight earlier tokens more heavily and have a
    256-token limit, so source code is included last and truncated if needed.
    """
    parts = []

    # Lead with the human-readable summary — best signal for cross-library matching
    if fn.get("comment"):
        parts.append(fn["comment"])

    parts.append(f"Function: {fn['qualified_name']}")

    if callees:
        parts.append("Calls: " + ", ".join(callees[:10]))

    if fn["source_code"]:
        code = fn["source_code"]
        if len(code) > MAX_CODE_CHARS:
            code = code[:MAX_CODE_CHARS] + "\n// ... (truncated)"
        parts.append(code)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS embedding index from code_graph.db")
    parser.add_argument("--db",         default=DEFAULT_DB,         type=Path, help="Path to code_graph.db")
    parser.add_argument("--index",      default=DEFAULT_INDEX,      type=Path, help="Output .faiss file")
    parser.add_argument("--meta",       default=DEFAULT_META,       type=Path, help="Output index_meta.json")
    parser.add_argument("--model",      default=DEFAULT_MODEL,                 help="HuggingFace model name or local path")
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int,  help="Embedding batch size")
    parser.add_argument("--project-root", default="",                          help="Strip this prefix from file paths in documents")
    parser.add_argument("--offline",    action="store_true",                   help="Set HF_HUB_OFFLINE=1 (use cached model only)")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # Load all function definitions (declarations have no source, less useful)
    functions = conn.execute("""
        SELECT f.id, f.qualified_name, f.display_name,
               fi.path AS file, f.line_start, f.line_end,
               f.kind, f.source_code, f.comment
        FROM functions f
        JOIN files fi ON fi.id = f.file_id
        WHERE f.is_definition = 1
    """).fetchall()

    print(f"Functions to embed: {len(functions)}")

    # Pre-load callee names per function  (caller_id -> [callee_name, ...])
    callee_map: dict[int, list[str]] = {}
    for row in conn.execute("SELECT caller_id, callee_qualified_name FROM calls"):
        callee_map.setdefault(row[0], []).append(row[1])

    conn.close()

    # Build one document per function
    documents = []
    metadata  = []
    for fn in functions:
        fn_id    = fn["id"]
        doc      = build_document(
            dict(fn),
            callee_map.get(fn_id, []),
            args.project_root,
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
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"Embedding {len(documents)} documents in batches of {args.batch_size}...")
    embeddings = model.encode(
        documents,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors -> cosine sim via dot product
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)

    # Build FAISS index (IndexFlatIP = exact inner-product = cosine on unit vecs)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(args.index))
    args.meta.write_text(json.dumps(metadata, indent=2))

    print(
        f"\nIndex written to  {args.index}  ({index.ntotal} vectors, dim={dim})\n"
        f"Metadata written to {args.meta}"
    )


if __name__ == "__main__":
    main()
