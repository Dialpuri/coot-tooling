"""Phase 2 — reconstitute verified gemmi ports into the Coot source tree.

Each phase-1 port is a standalone gemmi function (`coot::foo_gemmi(gemmi::Structure&,
args)`) with a passing test. This package lifts each port into a method of a new
parallel class `coot::<class>_gemmi` (holding a `gemmi::Structure structure;`
member), re-verifying the lift with the port's own adapted test, then merges the
verified methods into `<stem>_gemmi.{hh,cc}` files mirroring the originals.

See Phase2-Plan.md at the repo root for the full design.
"""
