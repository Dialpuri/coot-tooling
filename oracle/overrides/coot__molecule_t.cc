// coot::molecule_t instances are owned and managed by molecules_container_t.
// Access them via the mc[] operator after loading with read_pdb().
molecules_container_t mc;
mc.geometry_init_standard();

int imol = mc.read_pdb("@TEST_DATA_DIR@/example.pdb");
// mc[imol] is the coot::molecule_t for that molecule
