# Sample Protein-Ligand Benchmark Targets

This folder contains curated, ready-to-run protein-ligand targets for testing the drug generation pipeline.

### Target Catalog & Ready Coordinates

| PDB ID | Target Name | Co-crystallized Drug / Ligand | Pocket Center $(X, Y, Z)$ Å | Pocket Radius | Receptor PDB | Bound Reference Ligand |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- |
| **`2HYY`** | **Abl Kinase** | Imatinib (Gleevec) [`STI`] | `(14.251, 15.283, 17.632)` | 10.0 Å | `receptors/2HYY_receptor.pdb` | `ligands/2HYY_STI_bound_reference.pdb` |
| **`1HVR`** | **HIV-1 Protease** | Flap-binding Inhibitor [`XK2`] | `(-9.192, 15.906, 27.946)` | 10.0 Å | `receptors/1HVR_receptor.pdb` | `ligands/1HVR_XK2_bound_reference.pdb` |
| **`1UY6`** | **Hsp90 Chaperone** | PU3 Purine Inhibitor [`PU3`] | `(3.400, 11.180, 24.857)` | 10.0 Å | `receptors/1UY6_receptor.pdb` | `ligands/1UY6_PU3_bound_reference.pdb` |
| **`4COX`** | **COX-2 Synthase** | Indomethacin [`IMN`] | `(24.849, 22.295, 15.126)` | 10.0 Å | `receptors/4COX_receptor.pdb` | `ligands/4COX_IMN_bound_reference.pdb` |
| **`1M17`** | **EGFR Kinase** | Erlotinib [`AQ4`] | `(22.014, 0.253, 52.794)` | 10.0 Å | `receptors/1M17_receptor.pdb` | `ligands/1M17_AQ4_bound_reference.pdb` |
| **`1FKB`** | **FKBP12** | Rapamycin [`RAP`] | `(3.109, 12.167, 13.353)` | 10.0 Å | `receptors/1FKB_receptor.pdb` | `ligands/1FKB_RAP_bound_reference.pdb` |
| **`6LU7`** | **SARS-CoV-2 Mpro** | N3 Peptidomimetic [`02J`] | `(-10.464, 4.104, 73.010)` | 10.0 Å | `receptors/6LU7_receptor.pdb` | `ligands/6LU7_02J_bound_reference.pdb` |

### How to Run via CLI

```powershell
python run_full_pipeline.py `
  --pdb sample_examples/receptors/2HYY_receptor.pdb `
  --pocket_center 14.251 15.283 17.632 `
  --pocket_radius 10.0 `
  --k_states 2 `
  --n_mol 10 `
  --device cuda
```
