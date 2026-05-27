// Bond_lines_container is in the GLOBAL namespace (the surrounding
// `namespace coot {` in coords/Bond_lines.hh closes BEFORE the class).
// Do NOT write `coot::Bond_lines_container` — that does not exist and
// will fail to compile with "no type named 'Bond_lines_container' in
// namespace 'coot'".
//
// Required header: "coords/Bond_lines.hh"  (NOT in a subdir of "coot/")
//
// Bond_lines_container has many constructors (A–O in Bond_lines.hh).
// Pick by what your target function needs:
//
//   Constructor A — the default workhorse for "give me bonds for a molecule":
//      Bond_lines_container bonds(asc, imol);                 // simplest
//      Bond_lines_container bonds(asc, imol, 0, 1, false, false);
//
//   Constructor B — full bond-by-dictionary (used by moving-atoms graphics).
//      Needs a populated coot::protein_geometry *.
//
//   Constructor M — used by make_colour_by_chain_bonds() and friends.
//      Bond_lines_container bonds(geom_p, no_bonds_to_these_atoms, draw_h_flag);
//
//   Constructor O — empty default; you must call a do_*_bonds() method after.
//      Bond_lines_container bonds;
//
// You need a populated atom_selection_container_t. The fastest path is
// molecules_container_t → mmdb::Manager → coot::make_asc():

#include "coot-utils/atom-selection-container.hh"   // make_asc, atom_selection_container_t
#include "coords/Bond_lines.hh"                     // Bond_lines_container

molecules_container_t mc;
int imol = mc.read_pdb("@PDB_PATH@");

mmdb::Manager *mol = mc.get_mol(imol);
atom_selection_container_t asc = coot::make_asc(mol);   // make_asc is in `coot::`

// ─── Constructor A (simplest) ─────────────────────────────────────────────
Bond_lines_container bonds(asc, imol);

// ─── Constructor B (bond-by-dictionary, needs protein_geometry) ───────────
//   coot::protein_geometry geom;
//   geom.init_standard();
//   std::set<int> no_bonds;
//   Bond_lines_container bonds(asc, imol, no_bonds, &geom,
//                              /*include_disulphides*/ 1,
//                              /*include_hydrogens*/   1,
//                              /*draw_missing_loops*/  false,
//                              /*model_number*/        1,
//                              /*dummy*/               "",
//                              /*do_rama_markup*/      false,
//                              /*do_rota_markup*/      false,
//                              /*do_sticks_for_waters*/true,
//                              /*tables_p*/            nullptr);
//
// ─── Constructor M (colour-by-chain entry point) ──────────────────────────
//   coot::protein_geometry geom;
//   geom.init_standard();
//   std::set<int> no_bonds_to_these_atoms;
//   Bond_lines_container bonds(&geom, no_bonds_to_these_atoms, /*draw_h*/ true);
//   bonds.do_colour_by_chain_bonds(asc, false, imol, true, false, false, false, false);
//
// After construction, the typical follow-up is:
//   graphical_bonds_container gbc = bonds.make_graphical_bonds();
//
// ─── Inspecting state for assertions ──────────────────────────────────────
// `bonds` exposes (with -fno-access-control on oracle.cc):
//   bonds.bonds              // std::vector<Bond_lines>  — per-colour bond sets
//   bonds.atom_centres       // std::vector<...>         — populated by add_atom_centres
//   bonds.no_bonds_to_these_atoms
// For totals:
//   int total = 0;
//   for (const auto &bl : bonds.bonds) total += bl.size();
//
// ─── A note on coot::molecule_t's internal container ──────────────────────
// `coot::molecule_t::graphical_bonds_container_` is a
// graphical_bonds_container (the output of make_graphical_bonds()), NOT a
// Bond_lines_container. Do not try to grab the Bond_lines_container out of
// a coot::molecule_t — construct your own as shown above.
