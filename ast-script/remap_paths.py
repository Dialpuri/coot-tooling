#!/usr/bin/env python3
"""
Rewrite the path prefix stored in code_graph.db so the DB is portable
across machines where the coot source tree lives at a different location.

Usage:
  python ast-script/remap_paths.py \
      --old /Users/dialpuri/lmb/coot \
      --new /cluster/home/user/coot \
      [--db ast-data/code_graph.db]

After running this, set COOT_ROOT to the new prefix before invoking
the tooling so compiler flags and include paths resolve correctly:
  export COOT_ROOT=/cluster/home/user/coot
"""
import argparse
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).parent.parent / "ast-data" / "code_graph.db"


def remap(db_path: Path, old: str, new: str) -> None:
    old = old.rstrip("/")
    new = new.rstrip("/")

    conn = sqlite3.connect(db_path)
    before = conn.execute("SELECT COUNT(*) FROM files WHERE path LIKE ?", (f"{old}%",)).fetchone()[0]
    if before == 0:
        print(f"No paths starting with '{old}' found — nothing to do.")
        conn.close()
        return

    conn.execute(
        "UPDATE files SET path = ? || SUBSTR(path, ?) WHERE path LIKE ?",
        (new, len(old) + 1, f"{old}%"),
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM files WHERE path LIKE ?", (f"{new}%",)).fetchone()[0]
    print(f"Remapped {after} paths: '{old}' → '{new}'")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", required=True, help="Current path prefix in the DB")
    parser.add_argument("--new", required=True, help="Replacement path prefix")
    parser.add_argument("--db",  default=DEFAULT_DB, type=Path, help="Path to code_graph.db")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"DB not found: {args.db}")

    remap(args.db, args.old, args.new)


if __name__ == "__main__":
    main()
