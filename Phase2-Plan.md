# Phase 2 — Reconstituting Gemmi Ports into the Coot Source Tree

## Context

Phase 1 produced, per function, a verified gemmi reimplementation under
`generated-tests/coot__<class>__<method>/gemmi/` (`function.hh`, optional
`function.cc`, **and a passing gtest `test.cc`**). For `coot::molecule_t` alone
there are 177 such directories. The snippets are **structurally inconsistent** —
some are free functions in `namespace coot {}`, some are `static` methods inside a
fake `class molecule_t {}`, some sit in `namespace coot { namespace molecule_t {} }`,
some are declaration-only, many define helper structs/functions inline. In all of
them the original `molecule_t` member access (`atom_sel.mol`, an mmdb manager) was
replaced by an explicit `gemmi::Structure&` parameter so the function could be
tested standalone.

Phase 2 reconstitutes these verified ports into the **original Coot source tree**
(`/lmb/home/jdialpuri/Development/coot-dev/coot`) as **parallel files mirroring the
originals with a `_gemmi` suffix** — e.g. `api/coot-molecule.cc` →
`api/coot-molecule_gemmi.cc`. The functions are gathered into a new parallel class
that mirrors the MMDB one, so Coot can migrate call-sites incrementally.

gemmi is already a linked dependency (`-DUSE_GEMMI`, `find_package(GEMMI REQUIRED)`,
`target_link_libraries(cootapi PRIVATE ... gemmi::gemmi_cpp)`). Existing precedent
for gemmi code in the tree: `coot-utils/coot-coord-utils-gemmi.{hh,cc}`.

## Scope / generalization

**`coot::molecule_t` is the first target class**, but the infrastructure must apply
to **any class** (or any source file). Nothing in the design is molecule_t-specific:
the CLI takes a class qname or a source-file path, the DB grouping works for any
class, and the class-model/assembler derive the parallel-class name
(`<class>_gemmi`), header guard, and `_gemmi` filenames from the target. Hard-coded
`molecule_t` strings are forbidden — all naming flows from the target argument.

## Decisions (locked)

- **Target shape: a new parallel class `coot::<class>_gemmi`** (for the first
  target, `coot::molecule_t_gemmi`). NOT free functions. It mirrors the MMDB class.
- **Stateful:** the class holds `gemmi::Structure structure;` as a member. Ported
  methods drop the explicit `gemmi::Structure&` parameter and operate on
  `this->structure`, mirroring how the originals use `atom_sel.mol`.
- **Original method names:** drop the phase-1 `_gemmi` suffix — the class name
  already disambiguates. `coot::foo_gemmi(gemmi::Structure&, args)` →
  `coot::molecule_t_gemmi::foo(args)`.
- **The lift is done by an LLM and re-verified by the port's own test — NOT by
  regex.** A text/regex rewrite of C++ bodies is too fragile (inconsistent param
  names, aliasing, line-based parsing). Instead each port is lifted by an agent and
  the port's frozen-assertion test is adapted to construct the class and re-run.
- **Header layout: one header per class.** A single `coot-molecule_gemmi.hh`
  defines `class molecule_t_gemmi { ... };` declaring every ported method
  (mirroring the single original `coot-molecule.hh`). Each
  `coot-molecule-*_gemmi.cc` `#include`s it and defines the methods originally
  living in that `.cc`.
- **Commit gate:** a reconstituted file must compile **against the real coot
  include tree** before being git-committed.

## Approach

Two stages: a **per-port LLM lift that is verified by re-running the port's own
test** (mirrors phase 1's gemmi stage), then a **mechanical merge of the
already-verified fragments** into the parallel-class files, which are
build-registered, compiled against the coot tree, and committed.

The crucial property: the merge step is pure text assembly of pieces that already
pass a test, so the only semantic rewriting happens inside an agent loop that has a
green/red signal. There is no unverified mechanical body surgery anywhere.

### New module: `tooling/reconstitute/`

```
tooling/reconstitute/
  __init__.py
  classmodel.py      # derive <class>_gemmi name + fixed member set + construction
                     #   contract from the target; injected into every lift prompt
  agent.py           # per-port LLM lift; emits method (decl+def) + extra_members +
                     #   adapted test; verified by compile+run of that test
  generate.py        # orchestrates agent + compile + persist per-port artifacts
  merge.py           # assemble verified fragments -> <stem>_gemmi.hh + *_gemmi.cc
  build_registry.py  # idempotent CMake + Makefile.am registration
  commit.py          # write into coot tree + git commit (refactor of gemmi/commit.py)
  cli.py             # `python -m tooling.reconstitute <class-or-file>`
```

`LiftedMethod` dataclass: `qname, sig_hash, source_file, method_decl, method_def,
includes, helpers, extra_members, test_block, verified`.

### Step 1 — Class model (`classmodel.py`)

Derive the parallel class name (`<mmdb_class>_gemmi`) and FIX, up front, the state
and construction contract that every lift must target:
- members: `gemmi::Structure structure;` (seed), grown only by unioned
  `extra_members` from lifts;
- a default ctor + `explicit <class>_gemmi(gemmi::Structure s);` and a
  file/string factory (so adapted tests construct the object the same way every
  time).

This contract is rendered into a prompt fragment and injected into every per-port
lift so independently-lifted methods compose into one coherent class and the adapted
tests merge cleanly.

### Step 2 — Grouping (`tooling/db.py`)

Add helpers (reuse `get_function`/`get_function_overloads` overload-pinning, which
join `functions.file_id -> files.path`):

```python
def get_function_source_file(conn, qname, sig_hash=None) -> str | None:
    """Absolute path of the file that DEFINES this overload (prefer is_definition=1)."""

def group_entries_by_source_file(conn, entries) -> dict[str, list[tuple[str, str|None]]]:
    """Bucket passing (qname, sig_hash) by defining source file. Methods defined
    inline in the class header redirect to the sibling .cc stem so their
    definitions land in <stem>_gemmi.cc."""
```

Only reconstitute **passing** ports — reuse `tooling/batch.py`
`_is_complete` / `_gemmi_is_passing`.

### Step 3 — Per-port lift agent (`agent.py` + `generate.py`)

Reuse the `tooling/gemmi/generate.py` agent harness + `tooling/gemmi/compile.py`
(tier-1 gemmi+gtest env). For each passing port the agent receives:
- the port's `function.hh` / `function.cc`,
- its passing `test.cc`,
- the class contract from `classmodel.py`.

It produces, in labelled blocks (same mechanism as the gemmi agent):
- the **method declaration** (for the class body) and the out-of-line
  **definition** `<class>_gemmi::<name>(args)` that uses `this->structure`,
- any `extra_members` it needs (declared explicitly so merge can union them),
- an **adapted test**: constructs `<class>_gemmi` via the contract, sets/loads its
  structure, calls `obj.<name>(args)`, keeping the original **frozen assertions**.

Verification: compile + run the adapted test in a temp dir. **Pass = the lift is
correct** (same guarantee phase 1 had). Persist artifacts to a per-port
`generated-tests/coot__*/reconstitute/` dir (`method.hh`, `method.cc`, `test.cc`,
`agent_trace.txt`, a pass marker). Parallelizable across ports via the existing
worker/`OLLAMA_HOSTS` infra. Ports that never pass are reported, not dropped
silently.

### Step 4 — Merge verified fragments (`merge.py`)

Mechanical text assembly only (reuse `aggregate.py` `_split_header` /
`_dedup_ordered`). For a class:
- **`<stem>_gemmi.hh`**: license header, include guard `COOT_<STEM_UPPER>_GEMMI_HH`,
  deduped gemmi includes, `namespace coot { class <class>_gemmi { gemmi::Structure
  structure; <unioned extra members>; public: <ctors> <all verified method
  declarations>; private: <deduped helper decls> }; }`.
- **`<stem>[-X]_gemmi.cc`** (one per original `.cc`): `#include "<stem>_gemmi.hh"`,
  cc-only includes, the verified `<class>_gemmi::method` definitions for methods
  whose original lived in `X.cc`; helper definitions placed once (primary
  `<stem>_gemmi.cc`, anonymous namespace).
- **`<stem>_gemmi_test.cc`** (with `--with-tests`): concatenate the verified adapted
  `test_block`s, de-colliding `TEST(...)` names. They share the construction
  contract, so they compose into one binary.

Conflicting `extra_members` (same name, different type) across ports are flagged in
the report for human reconciliation.

### Step 5 — Build registration (`build_registry.py`)

Idempotent edits to BOTH build systems:

```python
def register_source_in_cmake(cmake_path, rel_src) -> bool   # add_library(... SHARED ...)
def register_source_in_makefile_am(mk_path, src) -> bool     # *_la_SOURCES (+ _HEADERS)
```

- CMake: insert `${coot_src}/<dir>/<stem>[-X]_gemmi.cc` into the matching
  `add_library(... SHARED ...)` list, matching existing indentation; idempotent
  substring check.
- Makefile.am: append gemmi sources into a delimited `# --- gemmi ports (phase 2)
  ---` block inside the relevant `*_la_SOURCES`; register `<stem>_gemmi.hh` in
  `_HEADERS`.
- Missing anchor = hard error (don't guess).

### Step 6 — Compile gate + commit (`commit.py`, `cli.py`)

- **Compile against the real coot tree:** reuse `COOT_BUILD_DIR` / ccp4 env from
  `tooling/oracle/compile.py` to `g++ -c <stem>[-X]_gemmi.cc` against coot's include
  closure. Catches composition errors the per-port tier-1 compile can't (type
  collisions with real coot types, member mismatches). A group must pass before
  commit.
- **Commit:** refactor `tooling/gemmi/commit.py` — extract git
  `add`/`commit`/`push` + `AGENT_COAUTHOR` logic into shared `_git_commit(repo,
  paths, msg)`; new layout writes `<src_dir>/<stem>[-X]_gemmi.{hh,cc}` instead of the
  old per-function `<src_dir>/gemmi/<sanitized>/` subdirectory.

### CLI (`cli.py`)

Mirror `batch.py`: positional `class_or_file`; `--filter`, `--list`/`--dry-run`,
`--workers N`, `--force` (re-lift), `--with-tests`, `--compile-tree` (default on,
the gate), `--register`, `--commit`. Per class: gather passing ports → group by
source file → per-port lift+verify (parallel) → merge verified fragments → write
`_gemmi` files → register → compile vs tree → optionally commit.

## Critical files

- Create: `tooling/reconstitute/{classmodel,agent,generate,merge,build_registry,commit,cli}.py`
- Modify: `tooling/db.py` (grouping helpers)
- Reuse: `tooling/gemmi/generate.py` + `gemmi/agent.py` + `gemmi/compile.py` (lift
  agent harness), `tooling/gemmi/aggregate.py` (`_split_header`, `_dedup_ordered`),
  `tooling/gemmi/commit.py` (git logic), `tooling/oracle/compile.py` (coot-tree
  compile env), `tooling/batch.py` (`_is_complete`/`_gemmi_is_passing`, worker pool),
  `tooling/oracle/generate.py` (`sanitize_name`).
- Coot tree targets (molecule_t example): `api/coot-molecule_gemmi.hh` (the class),
  `api/coot-molecule[-X]_gemmi.cc`, `CMakeLists.txt`, `api/Makefile.am`. Style
  reference: `coot-utils/coot-coord-utils-gemmi.{hh,cc}`.

## Verification

1. Lift one port end-to-end and confirm its adapted test passes — proves the agent
   contract works before scaling.
2. `python -m tooling.reconstitute "coot::molecule_t" --list` — confirm grouping by
   source file and which ports are already lift-verified.
3. Run a small group, inspect the generated `class molecule_t_gemmi` header + one
   `_gemmi.cc`, compile against the tree (no commit).
4. Scale to the full class; `coot-molecule.cc` (largest group, ~136 methods)
   assembles and compiles against the tree.
5. Build registration idempotent (re-run is a no-op); a real coot `make` picks up
   the new sources; `--with-tests` binary passes.
6. Re-run on a second class to confirm nothing is molecule_t-specific.
7. Only then `--commit`.

## Risks

- **Lift cost/throughput** — ~177 agent runs for molecule_t; mitigated by the
  existing parallel worker pool. Re-uses passing ports only, and `--force` re-lifts.
- **Class coherence across independent lifts** — the fixed member set + construction
  contract (`classmodel.py`) is what makes independently-lifted methods compose;
  conflicting `extra_members` are surfaced at merge for review. This replaces the
  old regex-fragility risk.
- **Class state creep** — uncontrolled `extra_members` growth makes `<class>_gemmi`
  a grab-bag; every introduced member is reported.
- **Makefile.am continuation editing** — mitigated by the delimited append block.
- **Composition errors invisible to per-port tests** — caught by the file-level
  compile-vs-coot-tree gate.
- **Large groups** (`coot-molecule.cc` ≈ 136 methods) stress merge/compile.
