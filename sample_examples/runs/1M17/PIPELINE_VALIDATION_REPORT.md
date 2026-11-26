# CMD-Gen Enhanced: Pipeline Run Report

**Target Receptor:** `1M17_receptor.pdb`  
**Pocket Center (Å):** `(22.01, 0.25, 52.79)`; radius `10.0 Å`  
**Pocket Center Source:** Sample run; RCSB PDB 1M17 ligand heavy-atom centroid  
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
| `G0` | Stage | Inputs & Applicability | **PASS** | `NONE` | PDB passes structural integrity and resolution checks (resolution: N/A A, 2497 atoms). |
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
| Anchor #1 | `HDON` | (21.52, 0.30, 53.61) | 100% | 1.320 |
| Anchor #2 | `HACC` | (21.91, -0.96, 53.77) | 100% | 1.213 |
| Anchor #3 | `HDON` | (22.25, 0.70, 54.16) | 100% | 0.419 |
| Anchor #4 | `HACC` | (20.10, 0.57, 53.69) | 100% | 0.188 |
| Anchor #5 | `HDON` | (23.44, 1.53, 56.29) | 100% | 0.109 |
| Anchor #6 | `HACC` | (24.80, -1.31, 52.11) | 100% | 0.000 |
| Anchor #7 | `HACC` | (24.06, 0.23, 50.95) | 100% | 0.000 |

---

## 3. Top-Performing Drug Candidates (Multi-State Docking)

| Rank | Candidate | Canonical SMILES | Mean $\Delta G$ (kcal/mol) | Worst-Case $\Delta G$ | Cross-State Variance |
|---|---|---|---|---|---|
| **#1** | `candidate_1` | `CCOC(=O)c1c2c(nn1-c1cc(C)n[nH]1)-c1ccccc1OC2(C)COC1CCCCC1` | **-8.38** | -8.38 | 0.000 |
| **#2** | `candidate_2` | `O=C(c1c(C2CCCCC2)n(-c2ccccc2OC(F)F)[nH]c(=O)c1=O)N1CCCCC1` | **-7.49** | -7.49 | 0.000 |

---

## 4. Interpretation and limitations

- This run reports model-generated pharmacophore features, generated structures, and Vina scores only.
- No comparative benchmark, off-target free-energy calculation, or wet-lab dose-response data were included. Gates requiring those data remain NOT EVALUATED.
- Vina scores are approximate docking rankings and do not establish affinity, selectivity, IC50, safety, or efficacy.