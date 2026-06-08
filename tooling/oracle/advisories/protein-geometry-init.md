---
name: protein-geometry-init
when: (?<![\w&*])protein_geometry\s+(?!&|\*)[A-Za-z_]\w*\s*[;{=]
unless: init_standard
---
ADVISORY (possible cause): a `coot::protein_geometry` is constructed here but `init_standard()` is never called. Without it the dictionary tables stay EMPTY — `get_monomer_restraints` / `get_group` / torsion and bond lookups then silently return empty/false (giving wrong or `XXXXXX`-style output) or segfault when they dereference the empty tables. If the failure above looks like an empty/garbage result or a crash inside geometry code, add `geom.init_standard();` immediately after constructing the protein_geometry and re-run. Ignore this if the failure is clearly unrelated to dictionary/restraint data.
