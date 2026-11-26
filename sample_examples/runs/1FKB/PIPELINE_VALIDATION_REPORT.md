# CMD-Gen Enhanced: Pipeline Run Report

**Target Receptor:** `1FKB_receptor.pdb`  
**Pocket Center (Å):** `(3.11, 12.17, 13.35)`; radius `10.0 Å`  
**Pocket Center Source:** Sample run; RCSB PDB 1FKB ligand heavy-atom centroid  
**Meeko Residue Handling:** unmatched residues may be omitted  
**Conformational Ensemble:** 1 aSAM-sampled state  
**Flow Matching Backbone:** Continuous Flow Matching (Optimal Transport, Euler integration)  
**Molecular Generator:** GCPG Conditioned Graph Transformer  
**Docking:** AutoDock Vina multi-conformation screen  

---

## 1. Checks and evidence status

PASS means the specific computational check ran and passed. NOT EVALUATED means required independent benchmark or experimental data were unavailable. Docking scores are computational prioritization scores, not binding free energies or potency measurements.

| Gate ID | Stage | Gate Name | Status | Action / Consequence | Details |
|---|---|---|---|---|---|
| `G0` | Stage | Inputs & Applicability | **PASS** | `NONE` | PDB passes structural integrity and resolution checks (resolution: N/A A, 1021 atoms). |
| `Q` | Stage | Ensemble QC | **PASS** | `NONE` | C-alpha RMSD/distance screen passed: 100.0% conformers passed (>= 80% required). |
| `G-LFO` | Stage | Family Generalization | **NOT EVALUATED** | `NONE` | Training/test family manifests are not present in this checkout. |
| `G2a` | Stage | PharmFlow vs. DiffPhar | **NOT EVALUATED** | `NONE` | No held-out labeled benchmark set or paired DiffPhar baseline scores were supplied. |
| `G1` | Stage | Pocket State Coverage | **NOT EVALUATED** | `NONE` | Independent experimental/reference pharmacophore states were not supplied. |
| `G2b` | Stage | Ensemble vs. Crystal Retrospective | **NOT EVALUATED** | `NONE` | No retrospective active/inactive ligand benchmark was supplied. |
| `G3` | Stage | Differential Feature Separability | **NOT EVALUATED** | `NONE` | No matched off-target structures or labeled feature benchmark was supplied. |
| `D-State` | Stage | State Ablation Specificity | **NOT EVALUATED** | `NONE` | No held-out retrieval benchmark was supplied for state ablation. |
| `G4` | Stage | Retrospective GO / NO-GO | **NOT EVALUATED** | `NONE` | No retrospective benchmark was supplied; generation is not evidence of prospective efficacy. |
| `FEP` | Stage | Selectivity Free Energy | **NOT EVALUATED** | `NONE` | Vina docking scores are not free-energy perturbation calculations; no off-target FEP data were supplied. |
| `G7-8` | Stage | Experimental Claim Verification | **NOT EVALUATED** | `NONE` | No measured dose-response or selectivity data were supplied. |

---

## 2. 3D Consensus Pharmacophore Blueprint

Generated **7 consensus pharmacophore features** from sampled structures. State agreement is reported per feature:

| Feature ID | Pharmacophore Type | Coordinates (X, Y, Z) | State Agreement | Interaction Dispersion |
|---|---|---|---|---|
| Anchor #1 | `HYBL` | (4.12, 8.01, 12.31) | 100% | 1.051 |
| Anchor #2 | `UNKNOWN` | (2.76, 8.35, 11.92) | 100% | 0.758 |
| Anchor #3 | `HYBL` | (3.64, 10.96, 12.43) | 100% | 0.846 |
| Anchor #4 | `HACC` | (3.16, 10.08, 10.57) | 100% | 0.428 |
| Anchor #5 | `HACC` | (2.48, 10.48, 13.68) | 100% | 0.796 |
| Anchor #6 | `HYBL` | (2.36, 6.03, 11.50) | 100% | 0.197 |
| Anchor #7 | `HYBL` | (1.39, 11.85, 12.00) | 100% | 0.370 |

---

## 3. Top-Performing Drug Candidates (Multi-State Docking)

| Rank | Candidate | Canonical SMILES | Mean $\Delta G$ (kcal/mol) | Worst-Case $\Delta G$ | Cross-State Variance |
|---|---|---|---|---|---|
| **#1** | `candidate_1` | `CCN=C(NCCc1ccc(F)cc1)c1nn(-c2cc(SC)c3ccccc3n2)c(C)c1C` | **-9.52** | -9.52 | 0.000 |
| **#2** | `candidate_2` | `CCCCc1cnc2c(c(-c3cccc(C)c3)cc(=O)n2-c2nc(S)c(C#N)c(CC)c2C#N)c1N` | **-8.07** | -8.07 | 0.000 |

---

## 4. Interpretation and limitations

- This run reports model-generated pharmacophore features, generated structures, and Vina scores only.
- No comparative benchmark, off-target free-energy calculation, or wet-lab dose-response data were included. Gates requiring those data remain NOT EVALUATED.
- Vina scores are approximate docking rankings and do not establish affinity, selectivity, IC50, safety, or efficacy.