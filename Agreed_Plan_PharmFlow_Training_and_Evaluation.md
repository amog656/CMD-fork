# Agreed plan: PharmFlow training, aSAMt use and evaluation order

Context file for the AI that will write the code. Read this together with `Implementation_Specification_V13.3_rev2.md`. If the two ever disagree, stop and ask the user. Do not choose silently.

## 1. What the project is

A target-configurable pipeline for pharmacophore-guided molecule design, built on the CMD-GEN paper (Zou et al., Communications Biology 8:422, 2025).

Fixed workflow, never reordered:

protein -> aSAMt ensemble -> QC -> pocket states -> PharmFlow -> consensus/differential -> frozen GCPG -> early filters -> scoring funnel -> MD + wet lab

- **PharmFlow is the only model we train.** It is a pocket-conditioned flow-matching model that replaces CMD-GEN's diffusion pharmacophore sampler (called DiffPhar here).
- **aSAMt is frozen.** It only supplies conformational states at inference time.
- **GCPG is frozen.** Use the released CMD-GEN code and weights. Never retrain it.
- The project is called target-configurable, not universal. "Universal" may only be claimed if held-out-family and multi-family evidence supports it.
- PARP1/PARP2 is one worked example, never a hard-coded assumption.

## 2. Decisions we have agreed

### 2.1 Training data for PharmFlow

1. **Train PharmFlow first on CrossDocked alone** (crystal pockets with real pharmacophore labels from the bound ligands), as CMD-GEN did. This is Model A and the baseline.
2. **Do not train on aSAMt outputs alone.** Every label comes from a crystal ligand. An aSAMt conformer has no ligand of its own, so its label would be copied from a crystal complex and may no longer fit the shape. Crystal data keeps the labels honest.
3. **Training on CrossDocked plus aSAMt conformers (Model B) is an optional second experiment**, run only if time allows, and only to test whether it beats Model A. It is not the default.
4. **aSAMt is added after training, at inference.** Its states go through QC, alignment, clustering and the domain filter (Step 3) before reaching PharmFlow.

### 2.2 Evaluation order

1. PharmFlow versus DiffPhar on crystal pockets (gate G2a).
2. PharmFlow on held-out protein families (gate G-LFO).
3. Ensemble versus crystal on held-out actives (gates G2b and G4).

Steps 1 and 2 must not depend on aSAMt, so any failure there is a PharmFlow problem and not an aSAMt problem.

### 2.3 Distribution shift is the main risk

PharmFlow is trained on crystal (mostly holo) pockets. aSAMt states are generated and may differ (shifted side chains, looser pockets, sampling artifacts, apo-like shapes). The model still returns an answer on unfamiliar input, with no warning. The defence is the **Step 3 domain filter**:

- Compute the frozen correspondence-free pocket distance (spec Section 10, C3) from each aSAMt representative state to the nearest PharmFlow training pocket.
- Pass the state to PharmFlow only if the distance is at or below the OOD cutoff.
- The OOD cutoff is the 95th percentile of nearest-neighbour distances measured on family-held-out calibration folds from CrossDocked only.
- **Never set or loosen the cutoff using aSAMt states.** That is tuning on the thing being tested.
- Log the rejected fraction. A large rejected fraction is a legitimate result and means the ensemble adds little.
- Out-of-domain states go to the Mode D loop (spec Section 10, C2), which means a new frozen model version, recalibrated cutoff, and reruns of G-LFO and G2a. Until then they are exploratory only.

## 3. How the 10 aSAMt states become molecules

(Example numbers only. K is set in the config and fixed.)

1. aSAMt gives about 10 states. QC drops broken ones (for example 8 remain).
2. States are aligned into one common frame and clustered. One representative per cluster is kept (for example 5), then the domain filter is applied.
3. **PharmFlow runs once per in-domain representative state.** It never receives several states at once. For each state it samples several pharmacophore point clouds (default 5 repeats), pools them, and clusters them with a Gaussian mixture model into one pharmacophore model per state.
4. **Merge the per-state models:**
   - Consensus model: pool points, group same-type points within 1.5 angstrom, keep groups supported by at least 50% of states (vote fraction 0.5). Different feature types never merge. Votes count distinct states.
   - Differential model (only if off-targets exist, m >= 1): S(f) = P_T(f) minus max over off-targets of P_Oi(f). Skipped when m = 0.
   - Optional state-specific models for features seen in only one or two states.
   - Vote fraction and grouping distance are pilot-chosen defaults. Freeze them before testing.
5. Gate G4 runs here, before generation.
6. GCPG runs on each chosen pharmacophore model (one hypothesis per run), with property gate values from the config.
7. Pool and filter molecules, then test every molecule against all K states (same K for all compounds and proteins).

## 4. Two different kinds of gating. Do not mix them up.

**Property gating (GCPG input).** User-set target values (MW, LogP, QED, SA score, rotatable bonds, TPSA) that steer generation. They are requests, not guarantees. Values come from rules of thumb (Lipinski, Veber, rule of three), the target's known actives (for example 10th to 90th percentile from ChEMBL), and project needs. Pick them from a pilot, record the source in the config, and freeze them before test evaluation. Always measure the achieved values with RDKit afterwards.

**Validation gates (checkpoints).** Defined in the spec. Outcomes are only PASS, FAIL and NOT_EVALUABLE, and NOT_EVALUABLE has the same consequence as FAIL.

| Gate | Step | Consequence on FAIL or NOT_EVALUABLE |
| --- | --- | --- |
| G0 | 0 | STOP the target run |
| Q | 2 | STOP the ensemble branch |
| G1 | 3 | FALLBACK: unweighted states |
| G2b | 3 | FALLBACK: crystal structure baseline |
| G-LFO | 4 | DOWNGRADE: no family-generalization claim |
| G2a | 4 | DOWNGRADE: no PharmFlow-over-DiffPhar claim |
| G3 | 5 (m >= 1) | DOWNGRADE: no selectivity-feature claim |
| D-state | 5 (Mode D) | DOWNGRADE: no state-specificity claim |
| G4 | after 5, before 6 | STOP confirmatory generation (exploratory from crystal baseline only) |
| FEP | 8 | DOWNGRADE: no FEP-supported claim |
| G7-8 | 9 | DOWNGRADE: no experimental claim |

Precedence: STOP overrides FALLBACK, which overrides DOWNGRADE.

**Who sets values and who measures them:** the user sets property gate values and freezes validation thresholds. Evaluation values are measured by outside tools (RDKit, docking, MM-GBSA, FEP, MD, simple counting). The model never sets or scores its own values.

## 5. Training rules for PharmFlow (Model A)

- **Pocket input comes from the protein only.** Never include ligand atoms (that leaks the label).
- **Centre** all coordinates on the pocket centroid. Use random rotation and translation augmentation. Test equivariance numerically.
- **Label:** pharmacophore points extracted from the bound ligand, mapped to 8 classes (AROM, HYBL, POSC, NEGI, HACC, HDON, LHYBL, OTHERS). Write the mapping as one explicit table and unit-test it.
- **Loss:** flow matching with optimal-transport (Hungarian) pairing of noise points to data points, plus a type classification term and a point-count term. Never compute a loss in arbitrary point order.
- **Splits are made on leakage clusters** (frozen pocket distance, connected components kept intact), never on individual complexes. Need at least 10 training families and at least 5 held-out families, otherwise G-LFO is NOT_EVALUABLE. Never relax cutoffs after seeing results.
- **Exclusion before training:** remove held-out actives, close analogs (Bemis-Murcko scaffold match or ECFP4 Tanimoto at or above 0.70) and pharmacophore near-duplicates (set Tanimoto or one-to-one F1 at or above 0.80 within 1.0 angstrom).
- **DiffPhar baseline:** same encoder, data, splits and compute budget as PharmFlow. Record compute for both. If not matched, G2a is NOT_EVALUABLE.
- **Inference:** integrate the learned ODE in 10 to 20 steps (Euler or Heun, count Heun as two evaluations per step). Record NFE and wall-clock time.
- **F1 definition:** same-type one-to-one Hungarian matching within 1.0 angstrom, per pocket, macro-averaged. Bootstrap over independent proteins or groups, never over single points.
- Fix seeds, log config hash, save checkpoints with the config hash.

## 6. If we later train Model B (CrossDocked plus aSAMt conformers)

Only after Model A is finished and evaluated. Required steps:

1. Manifest with one row per conformer (protein id, seed type, temperature, conformer id, path).
2. Run QC (Step 2) and log reasons for every drop.
3. Align each conformer to its crystal complex using pocket backbone atoms, transfer the crystal ligand and its pharmacophore points into the conformer frame, save the transform.
4. **Clash check:** drop or flag conformers where the transferred ligand clashes beyond a configured cutoff. Skipping this attaches labels to pockets that cannot bind that ligand.
5. Use the same pocket residue set for every conformer of a protein.
6. Remove near-duplicate conformers per protein.
7. Split by family, with all conformers of a protein in the same split.
8. Cap or reweight per protein so that proteins with many conformers do not dominate.
9. Remember that 15k conformers from a limited number of proteins is not 15k independent samples. Report the number of distinct proteins and families.
10. Decide, and write in the config, how multiple ligands per CrossDocked pocket are handled.

Compare Model A and Model B with the same proteins, splits and compute, on: held-out families (G-LFO), fraction of aSAMt states passing the domain filter, F1 on aSAMt states of proteins with a crystal ligand, and ensemble versus crystal (G2b, G4). Rule: if B wins, use B; if they tie, use A; if A wins, report that aSAMt data hurt.

## 7. Rules the AI must follow

1. Do not reorder or add pipeline stages.
2. No hard-coded thresholds. Every number comes from the config (Target Configuration Card), with its source recorded.
3. Never invent APIs for aSAMt, PharmFlow or GCPG. Write adapters with mocks and a TODO. Copy GCPG's pharmacophore-to-input conversion from the CMD-GEN repository, and ask the user if anything is unclear.
4. Never exponentiate docking scores into Boltzmann weights. Never compare raw docking scores across proteins.
5. State weights stay unweighted unless the shared convergence rule (spec C5) passes at the reference temperature.
6. Held-out data never enters seeds, training, calibration, state definition or threshold choice.
7. Compute expected test values with code, never by hand.
8. Seeds, versions and config hash are logged for every run.
9. If something is ambiguous, stop and ask. Do not pick a value.

## 8. Open items the user still has to supply

- Which aSAMt interface and weights are used, and how the 15k conformers were generated (seed types, temperatures, number of distinct proteins).
- Exact CrossDocked version and split files in use.
- GCPG repository version, weights, and its pharmacophore input format and maximum hypothesis size.
- Which docking, MM-GBSA, FEP and MD tools are available.
- Pilot-chosen values: vote fraction, grouping distance, K, property gate values, non-inferiority margin, FEP shortlist rule.
- Reference actives, inactives and decoys for the first target, and the off-target list (if any).
- Compute budget.

## 9. One-line summary

Train PharmFlow normally on CrossDocked, evaluate it on crystal pockets and held-out families, then add aSAMt only at inference behind the domain filter, and let the gates (G2b, G4) decide whether the ensemble is actually worth using. Test aSAMt-augmented training only as a later, separate experiment.
