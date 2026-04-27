"""
Entry point: run oracle -> test -> gemmi for every function in a source file.

Usage:
  python -m tooling.batch_file src/coot/molecule.cc
  python -m tooling.batch_file src/coot/molecule.cc --agent --workers 4
  python -m tooling.batch_file src/coot/molecule.cc --list
"""
from .batch import main_file

if __name__ == "__main__":
    main_file()
