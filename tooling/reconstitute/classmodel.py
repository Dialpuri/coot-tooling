"""The parallel `<class>_gemmi` class model and its fixed construction contract.

Every per-port lift targets the SAME class shape so independently-lifted methods
compose into one coherent class and their adapted tests construct the object the
same way. `ClassModel` owns:

  * name derivation  (coot::molecule_t -> namespace `coot`, class `molecule_t_gemmi`)
  * the seed state    (`gemmi::Structure structure;`) + canonical constructors
  * the contract prompt fragment injected into every lift
  * the rendered class skeleton reused by the merge step

Nothing here is molecule_t-specific — all naming flows from the target class qname.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# The member that holds the gemmi model — the standalone ports' explicit
# `gemmi::Structure&` parameter becomes a reference to this.
STRUCTURE_MEMBER = "structure"


@dataclass
class ClassModel:
    """Describes the parallel gemmi class derived from an MMDB class qname."""

    mmdb_qname: str                       # "coot::molecule_t"
    namespace: str                        # "coot"
    mmdb_class: str                       # "molecule_t"
    extra_members: dict[str, str] = field(default_factory=dict)
    # name -> full declaration line, e.g. {"imol_no": "int imol_no = -1;"}.
    # Grown by lifts that need state beyond `structure`; surfaced in the report.

    @property
    def class_name(self) -> str:
        return f"{self.mmdb_class}_gemmi"

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}::{self.class_name}"

    def header_guard(self, stem: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]", "_", stem).upper().strip("_")
        prefix = "" if slug.startswith("COOT_") else "COOT_"
        return f"{prefix}{slug}_GEMMI_HH"

    # ── rendered C++ ──────────────────────────────────────────────────────────

    def member_decls(self) -> list[str]:
        """The class's data members, seed first then any unioned extras."""
        decls = [f"gemmi::Structure {STRUCTURE_MEMBER};"]
        decls.extend(self.extra_members[k] for k in sorted(self.extra_members))
        return decls

    def constructor_decls(self) -> list[str]:
        """Canonical constructors. Tests build the object from a loaded Structure."""
        return [
            f"{self.class_name}() = default;",
            f"explicit {self.class_name}(gemmi::Structure {STRUCTURE_MEMBER}_in)",
            f"    : {STRUCTURE_MEMBER}(std::move({STRUCTURE_MEMBER}_in)) {{}}",
        ]

    def render_skeleton(self) -> str:
        """The class definition with members + ctors but no methods.

        Used by the lift agent's compile harness so a method can be verified in
        isolation against the real class shape, and as the spine merge.py fills.
        """
        body = "\n".join(f"    {m}" for m in self.member_decls())
        ctors = "\n".join(
            f"    {c}" if not c.startswith("    ") else f"    {c}"
            for c in self.constructor_decls()
        )
        return (
            f"namespace {self.namespace} {{\n"
            f"class {self.class_name} {{\n"
            f"public:\n"
            f"{body}\n"
            f"\n"
            f"{ctors}\n"
            f"}};\n"
            f"}}  // namespace {self.namespace}\n"
        )

    # ── prompt fragment ───────────────────────────────────────────────────────

    def contract_prompt(self) -> str:
        """The fixed construction contract injected into every lift prompt."""
        return CONTRACT_TEMPLATE.format(
            qualified=self.qualified_name,
            cls=self.class_name,
            ns=self.namespace,
            member=STRUCTURE_MEMBER,
        )


CONTRACT_TEMPLATE = """\
# Target class: `{qualified}`

You are lifting a standalone, already-working gemmi function into a METHOD of a
new parallel class that mirrors the original MMDB class. The class is fixed:

    namespace {ns} {{
    class {cls} {{
    public:
        gemmi::Structure {member};           // holds the model
        {cls}() = default;
        explicit {cls}(gemmi::Structure s) : {member}(std::move(s)) {{}}
        // ... your method goes here ...
    }};
    }}

Rules for the lift:
1. The standalone function takes an explicit `gemmi::Structure&` (named `st`,
   `mol`, `structure`, ...) as its FIRST parameter. As a method, DROP that
   parameter and operate on the `{member}` member (`this->{member}`) instead.
2. Keep the ORIGINAL method name — drop the `_gemmi` suffix. The class name
   already distinguishes it from the MMDB version. So `foo_gemmi(...)` becomes
   `{cls}::foo(...)`.
3. Do NOT change any other parameter, the return type, or the logic. This is a
   mechanical re-homing of a verified function, not a rewrite.
4. If the method genuinely needs class state beyond `{member}`, declare it as an
   extra public member of the class and say so — but prefer not to.
5. The adapted test must construct the object from the SAME structure the
   original test loaded, e.g. `{ns}::{cls} m(st); ... m.foo(args) ...`, keeping
   every original assertion (expected values are frozen).
"""


def class_model_for(class_qname: str) -> ClassModel:
    """Derive a ClassModel from a class qualified name like `coot::molecule_t`."""
    if "::" in class_qname:
        namespace, cls = class_qname.rsplit("::", 1)
    else:
        namespace, cls = "", class_qname
    return ClassModel(mmdb_qname=class_qname, namespace=namespace, mmdb_class=cls)


def gemmi_filenames(source_file: str) -> tuple[str, str]:
    """('.../api/coot-molecule-maps.cc') -> ('coot-molecule-maps_gemmi.cc',
    'coot-molecule-maps_gemmi.hh'). The .hh stem strips any trailing topic so the
    whole class shares one header (handled by the caller); here we just suffix."""
    stem = Path(source_file).stem
    return f"{stem}_gemmi.cc", f"{stem}_gemmi.hh"
