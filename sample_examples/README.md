# Sample protein pockets

These two examples are downloaded from the [RCSB Protein Data Bank](https://www.rcsb.org/) and prepared for the DiffPhar app. Each receptor is protein chain A with waters and bound ligand removed. The `ligands/` files preserve the crystallographic ligand coordinates for visual reference. Pocket centers are the mean coordinates of the ligand heavy atoms in the original complex.

| PDB | Target and ligand | Resolution | Center (Å) | Files |
|---|---|---:|---|---|
| [1FKB](https://www.rcsb.org/structure/1FKB) | FKBP12 · rapamycin (RAP) | 1.70 Å | `(3.109, 12.167, 13.353)` | `receptors/1FKB_receptor.pdb`, `ligands/1FKB_RAP_bound_reference.pdb` |
| [1M17](https://www.rcsb.org/structure/1M17) | EGFR kinase · erlotinib (AQ4) | 2.60 Å | `(22.014, 0.253, 52.794)` | `receptors/1M17_receptor.pdb`, `ligands/1M17_AQ4_bound_reference.pdb` |

The Streamlit app includes both structures in **Choose a structure**. Select one and click **Generate and dock molecules**; the receptor and pocket center are filled in automatically. The model checkpoints and external tools still need to be installed as described in the project README.

To run the command line directly, substitute the desired receptor, center, and output directory:

```powershell
python run_full_pipeline.py --pdb sample_examples/receptors/1FKB_receptor.pdb --pocket_center 3.109 12.167 13.353 --pocket_radius 10 --out_dir sample_run
```

`source/` stores the downloaded original coordinate files. `6LU7.pdb` is retained there as an unprepared source download, but is not offered as an example: the downloaded coordinate records did not contain the expected N3 inhibitor, so no pocket center was assigned from it.

These structures are research examples, not recommendations or evidence that a generated molecule is safe or effective. The ligand centroid is a convenient starting center; inspect the pocket and run results before drawing conclusions.
