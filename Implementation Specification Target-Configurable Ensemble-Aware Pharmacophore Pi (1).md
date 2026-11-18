# Implementation Specification: Target-Configurable Ensemble-Aware Pharmacophore Pipeline (V13.3)

*A step-by-step build document for an AI code generator. Read Section 0 first. Every later section assumes it.*

## 0. How to use this document (instructions to the code-generating AI)

This document describes a research pipeline in enough detail to implement it. Follow these rules for the whole project.

1. **Implement what is written. Do not improve, simplify or reorder.** The stage order is fixed: Step 0 to Step 9. Gates are attached where stated.
2. **Never invent an API.** Three components are external and only partly known: aSAMt (conformational sampler), PharmFlow (the model we train, no existing code), and GCPG (from the CMD-GEN repository, github.com/zyrlia1018/CMD-GEN). For each, write a thin adapter class with a clear interface and a mock implementation for tests. Put a TODO where the real call goes. Do not guess function names from memory.
3. **No hard-coded thresholds.** Every number (cutoffs, counts, margins, weights) must be read from one configuration file (the Target Configuration Card, TCC, Section 2.4). The numbers in this document are defaults for that file.
4. **Gates never pass silently.** If the data needed for a gate is missing or below its minimum sample size, the gate returns NOT\_EVALUABLE, and NOT\_EVALUABLE has the same consequence as FAIL.
5. **Every function that uses randomness takes a seed argument.** Log the seed, library versions and config hash in every output folder.
6. **Every stage writes its outputs to disk and a manifest** (what went in, what came out, how many items were dropped and why). The next stage reads only those files. Do not pass hidden state between stages.
7. **Write tests as you go.** Each step below ends with acceptance tests. A step is not done until they pass.
8. **Leakage rules are code, not comments.** Held-out actives and their close analogs must be excluded by a function that is called and unit-tested (Section 5, Step 1 and Section 10).
9. **If something is ambiguous, stop and raise a clear error** (or ask), instead of choosing a value.

## 1. The pipeline in one page

Input: a Target Configuration Card (TCC) and a protein target. Output: a ranked, filtered list of candidate molecules, a record of which gates passed, and a statement of which claims are allowed.

```
TCC + target
  -> Step 0  Inputs and applicability            [G0]
  -> Step 1  aSAMt ensemble (frozen)              
  -> Step 2  QC and restrained relaxation         [Q]
  -> Step 3  Pocket states and domain routing     [G1, G2b]
  -> Step 4  PharmFlow (only trained component)    [G-LFO, G2a]
  -> Step 5  Consensus and differential features  [G3, D-state]
  -> (G4 retrospective go/no-go, runs here, BEFORE Step 6)
  -> Step 6  Frozen GCPG molecule generation
  -> Step 7  Early filters
  -> Step 8  Scoring funnel                        [FEP]
  -> Step 9  MD and wet lab                        [G7-8]
```

Plain-language flow of the data:

1. The protein gives N conformational states (for example 10).
2. Bad states are dropped (QC). Similar states are merged into representatives (for example 5).
3. PharmFlow runs once per representative state and produces a pharmacophore model (a small set of typed 3D points) for each.
4. The per-state models are merged into a consensus model and, if off-targets exist, a differential model.
5. A retrospective test (G4) checks, using known actives and decoys, that this ensemble approach beats the crystal-structure baseline. Only if it passes does generation proceed in confirmatory mode.
6. GCPG turns each pharmacophore model plus property settings into molecules (SMILES strings).
7. Filters remove bad molecules. A funnel docks, rescores and runs free-energy calculations on a shrinking list.
8. The best candidates go to MD and the lab.

What is trained and what is not: **PharmFlow is the only model trained in this project.** aSAMt and GCPG are frozen. RDKit, docking, MM-GBSA, FEP and MD are measuring tools.

## 2. Global conventions

### 2.1 Gate outcomes and consequences

One outcome vocabulary is used everywhere.

- `PASS`: criterion met; proceed under the preregistered branch.
- `FAIL`: criterion not met; apply the gate's consequence.
- `NOT_EVALUABLE`: required evidence or minimum sample size is absent; apply the same consequence as FAIL. Never treat as PASS.

Each gate has exactly one consequence type:

- `STOP`: G0, Q, G4.
- `FALLBACK`: G1, G2b.
- `DOWNGRADE`: G-LFO, G2a, G3, D-state, FEP, G7-8.

Precedence: STOP overrides FALLBACK, which overrides DOWNGRADE for the affected branch.

Implement a `GateResult` record with these fields: gate\_id, outcome, consequence\_type, statistic values (point estimate, confidence interval, p-value if any), sample sizes, threshold used, config hash, and a human-readable reason string. Write every GateResult to a JSON-lines log. Write a function `apply_consequence(gate_result, run_state)` that updates a single `RunState` object (see 2.3). Do not scatter consequence logic inside stage code.

### 2.2 Core data structures

Define these as typed classes (Python dataclasses or pydantic) in one module, and use them in every stage.

- `PocketState`: state\_id, source (apo-seeded, holo-seeded, ligand-bound, crystal), file path to coordinates, pocket atom array (N x 3 coordinates, element, residue name, residue id, atom name), per-atom feature matrix, QC flags, cluster\_id, weight (float or None), in\_domain (bool), provenance (seed id, temperature, sampler version).
- `PharmPoint`: xyz (3 floats, in the common aligned frame, in angstroms), type (one of 8 classes: AROM, HYBL, POSC, NEGI, HACC, HDON, LHYBL, OTHERS), optional confidence.
- `PharmModel`: model\_id, list of PharmPoint, origin (per-state, consensus, differential, state-specific), supporting state ids, vote count per point.
- `Molecule`: smiles, source\_model\_id, property dict (MW, LogP, QED, SAS, TPSA, rotatable bonds), flags (valid, passed filters), scores per state (docking), aggregated scores, MM-GBSA and FEP values with uncertainty.
- `RunState`: current mode, list of surviving states, flags (ensemble\_allowed, confirmatory\_generation\_allowed, claims\_allowed as a set), and the gate log.

All coordinates are in angstroms. All energies are in kcal/mol. State this in a constants module and never convert silently.

### 2.3 The RunState object

A single object that every stage reads and updates through `apply_consequence`. Typical flags: `target_run_stopped`, `ensemble_branch_stopped`, `use_unweighted_states`, `use_crystal_baseline_only`, `confirmatory_generation_allowed`, and a set of withdrawn claims (for example `family_generalization`, `pharmflow_over_diffphar`, `state_specific_engagement`).

### 2.4 The Target Configuration Card (TCC)

One YAML file, validated against a schema at start-up. Missing required fields raise an error. Example with defaults:

```yaml
target:
  name: PLACEHOLDER
  construct: {chain: A, residues: [1, 300]}
  pocket_source: residues   # residues | detector
  pocket_residues: []
off_targets: []             # list of {name, structure_path}; length m
mode: A                     # A: m=0 potency, B: m=1, C: m>1, D: cryptic/allosteric
reference:
  actives_path: null
  inactives_path: null
  decoys_path: null
selectivity_goal_fold: 10
property_gates:             # input conditions for GCPG, not validation
  MW: 400
  LogP: 3
  QED: 0.6
  SAS: 4
  rotatable_bonds: null
  TPSA: null
state_weight_policy: unweighted   # unweighted | reference_T
K_states: 5                       # fixed number of states used for scoring
fep_shortlist_size: 15
n_minima: {G1: 20, G2b: 20, G3: 30, G4_actives: 100, G4_decoys: 100}
merge: {vote_fraction: 0.5, group_distance_A: 1.5}
thresholds: {}                    # filled from the register in Section 10
seed: 0
```

Important: the mode is derived from `len(off_targets)` and the `mode` field must agree with it. Mode D may have m = 0 or m >= 1. If they disagree, raise an error.

### 2.5 Folder layout and manifests

```
run_<hash>/
  config/tcc.yaml  config/resolved_config.json
  step0/ step1/ ... step9/      # each has outputs + manifest.json
  gates/gate_log.jsonl
  report/summary.md
```

Each `manifest.json` records counts in, counts out, dropped items with reasons, seeds, versions.

### 2.6 Suggested module layout

`config/` (schema, loader), `data/` (classes), `steps/step0.py` to `step9.py`, `gates/` (one file per gate plus `consequences.py`), `stats/` (bootstrap, permutation, Holm, Wilson, Hungarian F1), `adapters/` (asamt.py, pharmflow.py, gcpg.py, docking.py, mmgbsa.py, fep.py, md.py), `metrics/` (RDKit properties), `tests/`, `run_pipeline.py`.

## 3. Mistakes that apply to every step

1. **Mixing coordinate frames.** Every state must be aligned into one common frame before any cross-state operation. Store the transform used. Test: apply a random rotation to one state and confirm the outputs are unchanged after alignment.
2. **Silent unit changes** (nm vs angstrom, kJ vs kcal). Use the constants module only.
3. **Hard-coded numbers.** Search the code for numeric literals in stage files before finishing.
4. **Passing a gate when data is missing.** Missing means NOT\_EVALUABLE.
5. **Tuning on the test set.** Thresholds, K, vote fraction and grouping distance are chosen from pilot data and frozen (Section 10). Add a `freeze()` step that writes a hash of all frozen values before any test evaluation, and make test-evaluation functions refuse to run if the hash is missing.
6. **Leakage.** Held-out actives, close analogs (Bemis-Murcko scaffold match or ECFP4 Tanimoto >= 0.70) and pharmacophore-level near-duplicates must never enter seeds, training, state definition or calibration. See the leakage module.
7. **Using random splits where family splits are required** (G-LFO).
8. **Comparing raw docking scores across proteins.**
9. **Exponentiating raw docking scores into Boltzmann weights.** Not allowed.
10. **Non-reproducible runs.** Seeds everywhere, deterministic flags where possible, version pinning.
11. **Overwriting outputs.** Each run writes to a new hashed folder.
12. **Catching exceptions and continuing.** Failures must be recorded as dropped items with a reason, never swallowed.

## 4. Step 0: Inputs and applicability \[G0\]

**Purpose.** Decide whether the pipeline can legitimately run on this target, and fix the mode.

**Inputs.** The TCC, the target structure files (crystal or predicted), reference actives, inactives and decoys (if provided), off-target structures.

**Process.**

1. Load and validate the TCC. Compute `m = len(off_targets)`. Check the mode against m (A needs m = 0; B needs m = 1; C needs m > 1; D allows either).
2. Check that the construct residues exist in the structure files and that the structure has resolved coordinates for the pocket.
3. Define the pocket. If `pocket_source` is `residues`, take atoms of those residues. If `detector`, call the pocket-detection adapter on each state later (Step 3); at Step 0 only check that a detector is available. Pocket residue mapping from target to each off-target must cover at least 95% of pocket residues (a sequence or structure alignment); if less, selectivity analysis for that off-target is NOT\_EVALUABLE.
4. Count reference data: number of actives, inactives, decoys, matched pairs (same endpoint and assay type). Compare with `n_minima`.
5. Validate the construct against the full-length reference where one is given.
6. Produce an applicability report listing what is present and what is missing.

**Gate G0 (STOP).** PASS if a usable structure, a definable pocket and ligand evidence exist per the TCC. FAIL or NOT\_EVALUABLE if the target structure, construct, pocket or ligand evidence needed is missing: stop the target run. If only optional evidence is missing (for example no decoys), do not stop; mark the dependent downstream gates (G4, G2b, G3) as NOT\_EVALUABLE in advance in RunState and record why.

**Outputs.** `resolved_config.json`, `applicability_report.json`, cleaned structure files, pocket definition, the G0 GateResult.

**Handoff.** Step 1 receives the target structure, pocket definition and the list of reference ligands to hold out.

**Common mistakes.** Allowing mode and off-target list to disagree. Treating missing optional data as a stop. Not recording which gates are pre-marked NOT\_EVALUABLE. Using different residue numbering between target and off-target without a mapping.

**Acceptance tests.** (a) A TCC with m = 1 and mode A raises an error. (b) A TCC with no structure returns G0 = NOT\_EVALUABLE and stops. (c) A TCC with no decoys continues but marks G4 NOT\_EVALUABLE.

## 5. Step 1: aSAMt ensemble (frozen)

**Purpose.** Generate many plausible conformations of the target (and off-targets) so that pharmacophores can reflect protein flexibility.

**Inputs.** Target structure, pocket definition, list of held-out ligands, config (number of samples, seed set, temperature).

**Process.**

1. **Leakage first.** Build the exclusion set: all held-out actives, plus every ligand within the analog rule (Bemis-Murcko scaffold match or ECFP4 Tanimoto >= 0.70), plus every ligand whose pharmacophore set matches a held-out active's pharmacophore set (set Tanimoto >= 0.80 or one-to-one feature match F1 >= 0.80 within 1.0 angstrom). Any structure that contains an excluded ligand must not be used as a seed. Implement `build_exclusion_set()` and `assert_no_excluded_seed(seeds, exclusion_set)`; call both.
2. Prepare seeds: apo-seeded and holo-seeded starting structures, with the ligand removed from holo seeds. Record which seed produced which sample.
3. Call the aSAMt adapter to generate N conformers per seed (N from config). Record the sampling temperature for every sample.
4. Save each conformer as a structure file with provenance in a table.

**Important scientific caveat to encode.** aSAMt samples may come from high-temperature settings (for example 320 to 450 K). They are not Boltzmann populations at the reference temperature. Therefore state weights stay `None` unless Step 3 validates them with the convergence rule (Section 10, C5). Never assign weights from sample counts.

**Outputs.** A folder of conformer files and `ensemble_table.csv` with columns: state\_id, seed\_id, seed\_type, temperature, file path, sampler version.

**Handoff.** Step 2 reads the table and files.

**Common mistakes.** Seeding from a held-out complex. Forgetting to strip ligands from holo seeds. Treating sample frequency as population. Losing the seed or temperature record. Generating off-target ensembles with different settings than the target (they must use the same procedure and the same N).

**Acceptance tests.** (a) `assert_no_excluded_seed` fails when a held-out structure is placed in the seed list. (b) Every output row has a seed id and temperature. (c) Target and off-target ensembles have equal N.

## 6. Step 2: QC and restrained relaxation \[Q\]

**Purpose.** Remove physically broken conformers and lightly relax the rest.

**Inputs.** Ensemble table and files.

**Process.**

1. For each conformer, check geometry: bond lengths and angles within normal ranges, no chain breaks, no severe atom clashes, sensible backbone dihedrals, energy not an outlier versus the ensemble.
2. Apply restrained relaxation (positional restraints on the backbone so the conformation is kept but clashes are resolved). Re-check after relaxation.
3. Compute pocket RMSD of the pocket backbone atoms after alignment to the reference pocket. Mark states with pocket RMSD > 2.0 angstrom as QC failures.
4. Compute the pass fraction over the ensemble.
5. Keep the QC record per state: which checks failed.

**Gate Q (STOP, ensemble branch).** PASS if at least 80% of conformers pass geometry and energy QC and pocket RMSD <= 2.0 angstrom. FAIL or NOT\_EVALUABLE: stop this ensemble branch; do not relabel states as passing; do not use the ensemble in later steps. Q concerns ensemble QC only. State coverage is a different gate (G1).

**Outputs.** `qc_table.csv`, surviving conformer files, Q GateResult.

**Handoff.** Step 3 receives only the surviving states.

**Common mistakes.** Mixing up Q (is the ensemble physically valid) with G1 (does it cover the reference states). Computing RMSD without aligning first. Relaxing so hard that states collapse to the same structure (check that the pairwise RMSD spread stays similar before and after relaxation). Not saving the reason for each rejection.

**Acceptance tests.** (a) A synthetic ensemble with 30% deliberately broken structures gives Q = FAIL. (b) Pocket RMSD of a structure to itself is 0. (c) Relaxation does not change the number of atoms.

## 7. Step 3: Pocket states and domain routing \[G1, G2b\]

**Purpose.** Turn the surviving conformers into a small set of distinct pocket states, each safe to feed to PharmFlow.

**Inputs.** Surviving conformers, pocket definition, the PharmFlow training-pocket library (for the domain test), reference states if any.

**Process.**

1. **Align** all conformers into one common frame using the stable pocket backbone atoms (or core residues). Save the transform for each.
2. **Extract the pocket** per state (TCC residues, or a detector run per state; detectors also support cryptic pockets in Mode D).
3. **Cluster** aligned pockets (for example agglomerative clustering on pocket-atom RMSD, or on the point-cloud distance of Section 10, C3). Choose a representative (medoid) per cluster. Retain rare clusters only if justified (for example they contain the reference holo state). Target about K\_states clusters; set K from the TCC and keep it fixed afterwards.
4. **Domain routing.** For each representative, compute the distance to the nearest PharmFlow training holo pocket using the frozen correspondence-free pocket distance (C3). If the distance <= the OOD cutoff (95th percentile of nearest-neighbour distances on family-held-out calibration folds, C4), mark `in_domain = True` and send to PharmFlow. Otherwise mark `in_domain = False` and route to the Mode D loop (Section 10, C2): ligand-bound structures or MD/enhanced-sampling data, then retraining and revalidation. Until that is done, out-of-domain states are exploratory only and must not feed confirmatory results. Log the rejected fraction.
5. **State weights.** Default is unweighted. Compute weights only if reference-temperature data exist and pass the convergence rule (C5). Otherwise `weight = None`.

**Gate G1 (FALLBACK).** PASS if at least 90% reference-state coverage within 1.5 angstrom (a reference state is covered if some retained state lies within 1.5 angstrom of it), and, where valid reference weights exist, weighted-versus-unweighted rank correlation rho >= 0.8. If there is no reference state, report descriptive coverage and mark G1 NOT\_EVALUABLE. FAIL or NOT\_EVALUABLE: use the unweighted state ensemble, forbid population-weighted free energies, and label state populations descriptive.

**Gate G2b (FALLBACK).** Per-target paired comparison of ensemble versus crystal baseline on held-out actives: AUC(ensemble) minus AUC(crystal), with a paired-bootstrap 95% confidence interval; PASS only if the lower bound > 0 and the n-minimum is met. Use identical held-out ligand groups in both arms. FAIL or NOT\_EVALUABLE: use the crystal-structure baseline and state that ensemble benefit is not established. Implementation note: G2b needs docking or scoring of held-out actives and decoys against both arms, so it is evaluated through the same retrospective machinery as G4 (Section 8.5).

**Outputs.** `states.json` (list of PocketState with cluster ids, in\_domain flags, weights), aligned pocket files, G1 and G2b GateResults, rejected-fraction report.

**Handoff.** Only `in_domain` representatives go to Step 4 (K of them). The full crystal baseline structure also continues as a comparison arm.

**Common mistakes.** Clustering before aligning. Keeping all conformers instead of representatives. Computing the OOD distance with residue-by-residue RMSD across different proteins (undefined: use the correspondence-free distance only). Choosing the OOD cutoff using test structures (use calibration folds only and freeze). Letting K vary between target and off-target runs. Assigning weights from cluster sizes.

**Acceptance tests.** (a) Rotated and translated copies of a state land in the same cluster. (b) A state with distance above the cutoff is flagged out-of-domain and does not reach Step 4. (c) With no reference state, G1 is NOT\_EVALUABLE and the run switches to unweighted. (d) K\_states is identical for target and off-target.

## 8. Step 4: PharmFlow, the only trained component \[G-LFO, G2a\]

**Purpose.** Given one pocket state, produce a pharmacophore model: a small set of typed 3D points showing where chemical features should sit. PharmFlow is a pocket-conditioned flow-matching model. It replaces the diffusion sampler used in CMD-GEN (called DiffPhar here). CMD-GEN's sampler used an equivariant denoising diffusion model with an EGNN noise predictor (6 angstrom cutoff, 256 dimensions, 5 layers, 500 diffusion steps). Flow matching is expected to give similar quality in far fewer sampling steps, and this must be tested, not assumed.

**Inputs (training).** A library of protein-ligand complexes (CMD-GEN used CrossDocked2020, filtered to ligand pose RMSD <= 2 angstrom, about 100,000 pairs). For each complex: the pocket and the holo-derived pharmacophore points of the bound ligand.

**Inputs (inference).** One in-domain PocketState from Step 3.

**Outputs.** One PharmModel per state.

### 8.1 Building the training data

1. **Pocket.** Take protein atoms within a cutoff of the bound ligand (cutoff in config). Heavy atoms only, or alpha-carbon only as an option (CMD-GEN tested both).
2. **Target pharmacophore.** Extract features from the bound ligand with a feature definition set (for example the RDKit feature factory) and map them to eight classes: AROM (aromatic), HYBL (hydrophobic), POSC (positive ionizable), NEGI (negative ionizable), HACC (acceptor), HDON (donor), LHYBL (lumped hydrophobe), OTHERS. Each point is a position plus a class. The mapping from feature factory families to these classes is an implementation decision: write it as one explicit table in code and unit-test it on a known ligand.
3. **Centering.** Subtract the pocket centroid from all coordinates. All learning happens in this centred frame, and the inverse shift is applied at inference.
4. **Family labels and leakage clusters.** Assign each complex to a leakage cluster using the frozen correspondence-free pocket distance (Section 10, C3 and C4). Splits are made on clusters, never on individual complexes.
5. **Exclusion.** Remove any complex whose ligand is a held-out active, a close analog (scaffold match or ECFP4 Tanimoto >= 0.70), or a pharmacophore near-duplicate (set Tanimoto >= 0.80 or one-to-one F1 >= 0.80 within 1.0 angstrom). Log how many were removed.
6. **Splits.** Training families, calibration folds (used to set the OOD cutoff and for early stopping), and held-out families (used only for G-LFO). The test-set structures must never touch calibration.

### 8.2 Model architecture

1. **Pocket graph.** Nodes are pocket heavy atoms. Edges connect atoms within the cutoff. Node features: atom type, residue type, physicochemical properties. Edge features: distance (and optionally angle and orientation).
2. **Equivariant encoder.** An SE(3)-equivariant network (EGNN or an SE(3)-Transformer) produces per-atom features and a pocket embedding c. Keep per-atom features; do not collapse the pocket into one vector too early, because cross-attention needs them.
3. **Joint point-pocket network.** The noisy pharmacophore points (coordinates plus type vector) attend to pocket atoms by cross-attention with equivariant interaction. Conditions: pocket embedding c, time t, current point state z\_t.
4. **Velocity head.** Outputs a velocity for each point: a coordinate velocity (which must be equivariant) and a type-space velocity.
5. **Point count.** The number of pharmacophore points is variable. Train a small count head (or an empirical distribution conditioned on pocket size) and sample N from it at inference.

### 8.3 Training objective

Define time t from 0 (noise) to 1 (data). For each training pocket with data points z1 (N points) and noise z0 drawn from a standard normal distribution:

- Pair noise points to data points with the Hungarian algorithm on coordinate distance (optimal-transport pairing, per sample).
- Interpolate: z\_t = (1 - t) \* z0 + t \* z1. The target velocity is z1 - z0.
- Loss: L = L\_OT + lambda\_type \* L\_type + lambda\_count \* L\_count, where L\_OT is the squared error between predicted and target velocity on coordinates, L\_type is a classification loss on point type (cross-entropy), and L\_count is a loss on the point count. lambda values come from config.
- Handle unordered points: the pairing step is what removes ordering. Never compute a loss between points in arbitrary index order.

Design decision that must be validated: types are carried as a continuous vector (for example a one-hot relaxation) along the flow and converted to a class by argmax at t = 1. If you choose a different treatment, write it down and test it, because this is a known place where implementations quietly break.

Training details: Adam optimizer, learning rate and batch size from config, random rotation and translation augmentation, early stopping on the calibration folds, fixed seeds, checkpoints saved with config hash. Log the training families used.

### 8.4 The DiffPhar baseline (needed for G2a and G4)

Implement DiffPhar as the same pipeline with the same encoder, data, splits and compute budget, differing only in the generative process (denoising diffusion with a 500-step noise schedule and an EGNN noise predictor, as in CMD-GEN). Record training compute for both. If the two are not trained under matched conditions, G2a is NOT\_EVALUABLE.

### 8.5 Inference for each state

1. Load the in-domain state and its pocket graph. Centre on its centroid.
2. Sample N from the count head. Draw z0 from a standard normal distribution.
3. Integrate dz/dt = v\_theta(z, c, t) from t = 0 to t = 1 with 10 to 20 steps (Euler or Heun; Dormand-Prince optional). Record the number of function evaluations (NFE) and wall-clock time. These feed G2a.
4. Convert type vectors to classes. Shift coordinates back to the common aligned frame.
5. Repeat R times (default 5, as in CMD-GEN) with different seeds and pool all points.
6. **Cluster the pooled points** with a Gaussian mixture model on coordinates. Within each cluster, assign the most probable feature type, taking into account the overall feature probability and its probability inside that cluster (CMD-GEN, eq. 6). Choose the number of mixture components with a fixed rule (for example BIC within a set range) and freeze that rule.
7. Output one PharmModel for the state, with each point carrying its cluster support.

### 8.6 Gates at Step 4

**F1 definition (used throughout).** Compare a generated pharmacophore with a reference pharmacophore. Match points one-to-one with the Hungarian algorithm, only between points of the same type, accepting a match only if the distance <= 1.0 angstrom. Unmatched generated points are false positives; unmatched reference points are false negatives. Compute F1 per pocket, then average across pockets (macro-average is primary; micro-F1 is secondary). Bootstrap over independent proteins or complexes or predefined blocks, never over individual points.

**G-LFO (DOWNGRADE).** Leave-family-out test. Using leakage clusters, train on at least 10 families and test on at least 5 held-out families. Compute delta = F1(held-out families) minus F1(in-family test). PASS if the Holm-adjusted one-sided lower confidence bound of delta is above minus 0.05 (non-inferiority margin 0.05 absolute F1). If clustering yields fewer than 10 training or fewer than 5 held-out families, return NOT\_EVALUABLE; never loosen cutoffs afterwards. Consequence: withdraw the family-generalization claim.

**G2a (DOWNGRADE).** PharmFlow versus DiffPhar on matched data. The TCC preregisters exactly one claim: either superiority (delta F1 = F1\_flow minus F1\_diff, PASS if Holm-adjusted one-sided p < alpha and delta > 0) or non-inferiority (PASS if delta > minus delta\_margin, with delta\_margin <= 0.05, chosen from an independent pilot and frozen before test access). For a speed claim, also report median wall-clock and NFE under the same hardware. Consequence: withdraw the PharmFlow-over-DiffPhar claim.

**Common mistakes at Step 4.**

- Comparing point sets in index order. Always match with Hungarian assignment.
- Allowing the model to see absolute coordinates. Always centre; test equivariance.
- Training and testing families that overlap through near-identical pockets. Always cluster first.
- Choosing the non-inferiority margin after seeing test results. Freeze it.
- Comparing PharmFlow trained for longer or with a different backbone than DiffPhar.
- Counting NFE incorrectly (Heun uses two evaluations per step).
- Using atoms from the ligand as pocket input (leaks the answer). Pocket must come from the protein only.
- Forgetting the type argmax and then reporting soft types as classes.
- Evaluating F1 on training pockets.

**Acceptance tests.** (a) Rotating a pocket rotates the generated points by the same rotation (within numerical tolerance). (b) A pocket-free random point generator scores clearly lower F1 than the trained model on the calibration folds. (c) Hungarian F1 equals 1.0 when generated equals reference, and 0 when types never match. (d) The training set contains no excluded ligand (assert). (e) DiffPhar and PharmFlow runs record identical data splits.

## 9. Step 5: Consensus and differential features \[G3, D-state\], then the G4 go/no-go

**Purpose.** Combine per-state pharmacophore models into the models that will drive generation, and, when off-targets exist, find features that favour the target.

**Inputs.** The per-state PharmModels of the target (K states); if m >= 1, the per-state PharmModels of each off-target, generated with the same procedure, the same K and the same number of repeats; the common aligned frame; config merge parameters (vote fraction 0.5, grouping distance 1.5 angstrom).

### 9.1 Consensus model

1. Pool all points from all K state models. All points are already in one frame.
2. Group points of the same type that lie within the grouping distance of each other (for example single-linkage or a fixed-radius connected-components rule; choose one, freeze it). Points of different types never merge.
3. Vote count for a group = number of distinct states contributing at least one point to it. Do not count the same state twice.
4. Keep groups with vote fraction (votes / K) >= the vote fraction in config. If state weights were validated, use weighted votes; otherwise equal weights.
5. The group position is the mean of its points; the type is the group's type. Output the consensus PharmModel with votes recorded on each point.
6. Optionally output state-specific models: groups supported by one or two states only, kept as separate hypotheses for generation (useful for cryptic or rare-state features). Label them as state-specific.

### 9.2 Differential model (only if m >= 1)

1. Align each off-target pocket into the target frame using a structure-based pocket alignment. If the pocket residue mapping covers less than 95%, that off-target is NOT\_EVALUABLE.
2. For every consensus-candidate feature f in the target frame, estimate its presence probability in each protein X as P\_X(f) = (number of states of X that have a same-type feature within 1.5 angstrom of f) divided by (number of states of X).
3. Selectivity score S(f) = P\_T(f) minus the maximum over off-targets of P\_Oi(f). The maximum makes the weakest margin govern when m > 1. At m = 1 it reduces to a pairwise comparison.
4. Keep features with S(f) at or above the configured threshold (default 0.30) and with a bootstrap confidence interval lower bound above 0 (bootstrap over states). Freeze the threshold.
5. Exclusion regions: space occupied by off-target features that the target lacks. GCPG does not accept exclusion volumes, so these are applied later as a post-generation filter (Step 7), not as input to GCPG. Do not pretend GCPG consumes them.
6. If m = 0, skip this section entirely and do not compute any selectivity quantity.

### 9.3 Gate G3 (DOWNGRADE, m >= 1)

Differential-feature separability. Build per-feature vectors with target or off-target labels. Because there are only two proteins when m = 1, a leave-one-protein-out split is impossible (one class would vanish). Use blocked cross-validation with blocks defined by replicate or seed crossed with state cluster, and permute labels within matched blocks for the null. Report AUC and the permutation p-value. PASS if AUC >= 0.75 (effect-size target) and permutation p < 0.01. Below the n-minimum: NOT\_EVALUABLE. Consequence: withdraw the differential-feature separability claim. Write in the report that this is not a leave-protein-out test.

### 9.4 Gate D-state (DOWNGRADE, Mode D only)

State specificity for allosteric or cryptic design. Ablate state information on held-out actives only (all test analogs excluded from training). PASS if potency AUC drops by at least 0.05 with a Holm-adjusted confidence lower bound above 0, and at least one orthogonal evidence modality is present: (i) a ligand-bound structure resolving the predicted state, (ii) hydrogen-deuterium exchange mass spectrometry showing preregistered site-localised differential protection, or (iii) a functional perturbation that stabilizes or disrupts the predicted state and changes the response in the predicted direction. For m = 0 this is the only state-specificity evidence. For m >= 1 it is in addition to G3. Consequence: withdraw the state-specific engagement claim.

### 9.5 Retrospective go/no-go G4 (STOP for confirmatory generation)

G4 runs after Step 5 and before Step 6. It decides whether the ensemble approach has earned the right to drive molecule generation.

**Arms.** Crystal x DiffPhar, Crystal x PharmFlow, Ensemble x DiffPhar, Ensemble x PharmFlow. The primary contrast (hypothesis H5) is Ensemble x PharmFlow versus Crystal x PharmFlow. The Ensemble x DiffPhar versus Crystal x DiffPhar contrast is secondary and explanatory only; do not treat it as a second primary test.

**Procedure.**

1. Use the held-out actives and matched decoys from the TCC. Decoys are property-matched (MW, LogP, hydrogen-bond donors and acceptors, charge, rotatable bonds) and must not be analogs of the actives.
2. For each arm, take its pharmacophore model(s). Dock every active and decoy into the arm's retained states with pharmacophore constraints, using the same K for every compound and the same aggregation rule (best state, with the state recorded; report median and range too).
3. Score each compound by its aggregated docking score (energy-like: more negative is better).
4. Compute AUC per arm, and the paired AUC difference (ensemble minus crystal) with a paired bootstrap confidence interval, resampling by ligand series or complex group, not by single compounds.
5. Correct for best-of-K selection inflation with the permutation null in Section 10, C6.

**Pass rule.** PASS if the AUC gain >= 0.05 and the Holm-adjusted one-sided test passes. EF at 1% is a secondary metric and only reported with at least 100 actives and at least 9,900 matched decoys (a 1:99 ratio gives a top 1% of 100 compounds); AUC requires at least 100 actives and at least 100 decoys. Below these sizes: NOT\_EVALUABLE.

**Consequence.** PASS: confirmatory Step 6 is allowed. FAIL or NOT\_EVALUABLE: confirmatory Step 6 is blocked; optional exploratory generation may proceed only from the crystal-structure baseline and every output is labelled exploratory, with no ensemble-benefit or retrospective-success claim.

**G2b** reuses this same machinery (ensemble versus crystal on held-out actives) and sits on Step 3. Build the retrospective scorer once, as one module, and use it for both gates.

**Common mistakes at Step 5 and G4.**

- Merging points of different types.
- Counting votes per point rather than per distinct state.
- Using different K or repeats for target and off-target models, which biases P\_T versus P\_O.
- Comparing docking scores across proteins to decide selectivity. Rank within one protein only.
- Bootstrapping individual compounds when compounds come in analog series (inflates confidence).
- Using decoys that are easier than actives (property mismatch) or that contain actives of other targets.
- Letting any held-out active or its analog appear in aSAMt seeds, PharmFlow training or state definition.
- Reporting EF at 1% with too few actives.
- Skipping the best-of-K null.
- Running G4 after generation. It must run before.

**Acceptance tests.** (a) A consensus built from five identical state models returns exactly that model with vote fraction 1.0. (b) With m = 0, no function from the differential module is called. (c) Shuffling target and off-target labels drives G3 AUC to about 0.5 and the permutation p-value high. (d) A scorer that assigns random scores gives AUC near 0.5 on the retrospective set. (e) If decoys are fewer than the minimum, G4 returns NOT\_EVALUABLE and confirmatory Step 6 is blocked in RunState.

## 10. Preregistered methods register (implementation of C1 to C14)

Every item below is a function or a rule that must exist in code. Values are defaults read from config.

**C1. Gate outcomes.** Implement the vocabulary and consequences of Section 2.1 and the precedence rule. Specific consequences: G0 FAIL or NOT\_EVALUABLE stops the target run; Q stops the ensemble branch; G4 blocks confirmatory Step 6; G1 FAIL forces unweighted states and forbids population-weighted free energy; G2b FAIL forces the crystal baseline; G-LFO, G2a, G3, D-state, FEP and G7-8 withdraw their named claims. Terms like exploratory, unresolved and indeterminate are labels, not gate outcomes: unresolved FEP maps to NOT\_EVALUABLE for that claim; indeterminate is a retrospective label.

**C2. Mode D out-of-domain loop.** Cryptic or out-of-distribution states may use ligand-bound experimental structures or MD or enhanced-sampling data to expand training, only if the data is not derived from held-out test complexes and not seeded with held-out ligand structures. A new frozen PharmFlow version is created, the OOD distance is recalibrated, and G-LFO and G2a are rerun on that version. Held-out actives, analogs and pharmacophore-overlap matches stay excluded. If revalidation is impossible, the Mode D result is exploratory only. Implement a model-version registry so no result is reported without its model version.

**C3. Pocket distance.** Represent each pocket as a point cloud in a local coordinate frame with channels: atom element (weight 0.25), residue or chemical class (0.25), donor, acceptor, charge, aromatic, hydrophobic features (0.30), normalised occupancy or density (0.20). Distance D = sum over channels of w\_c times D\_c plus lambda times M, where each D\_c and the mismatch fraction M are normalised to \[0, 1\], lambda = 0.10. D\_c is a symmetric Chamfer distance after rigid alignment. Freeze preprocessing, weights and metric before calibration. Never use residue-by-residue RMSD across unrelated proteins. Test: D(x, x) = 0, symmetry, and invariance to rotation and translation.

**C4. OOD cutoff and leakage feasibility.** The OOD cutoff is the 95th percentile of nearest-training-holo-pocket distances computed on family-held-out calibration folds only. Leakage graph: nodes are proteins, edges join pockets within the frozen distance threshold; connected components stay intact within one split. If fewer than 10 training or 5 held-out families result, G-LFO is NOT\_EVALUABLE. No post-hoc relaxation.

**C5. D-state evidence and shared weight convergence.** One function, `weights_converged()`, used by both G1 and FEP. Converged only if: at least 3 independent reference-temperature replicas; split-half population difference <= 0.10 per state; rank correlation >= 0.80 across replica estimates; block-bootstrap 95% confidence interval width <= 0.20 per state. If unmet, G1 is unweighted and descriptive, and FEP uses the worst-case state-specific result. Never write two copies of this logic.

**C6. Fixed K and best-of-K null.** The TCC fixes K and the state-selection rule before scoring. Use the same K for all compounds and all proteins. Report best-state identity, median, range and K. Null: within each target, permute compound-to-state score assignments independently within each of the K state slots across compounds, preserving each slot's score distribution; recompute the maximum over K for every permutation to get an empirical selection-adjusted p-value. Use identical K and aggregation in G4 and FEP.

**C7. Docking sign and scope.** For energy-like scores (more negative is better): delta\_S\_dock = score(O\_i) minus score(T); positive favours T. Use docking only for within-protein ranking and calibrated triage. Never compare raw docking scores across proteins or apply cross-protein differences as hard filters. If a docking program returns positive-is-better scores, convert at the adapter boundary and test the sign.

**C8. Holm family.** Primary hypotheses fixed before evaluation: H1 G-LFO (family-level recovery; cluster permutation), H2 G2a (protein or family level paired pocket-macro-F1; grouped permutation), H3 G2b (per-target paired AUC; ligand-series or complex paired test), H4 G3 (block-level label permutation), H5 G4 (per-target paired AUC gain; paired ligand-series or complex test). Include every target-specific instance. Convert to one-sided p-values; apply the Holm step-down procedure at familywise error rate 0.05; invert confidence intervals from the Holm-adjusted tests; ordinary 95% intervals are descriptive only. Confirmatory family-level inference needs at least 10 independent groups, or a preregistered simulation-based power analysis showing at least 80% power at the target effect. Otherwise label results descriptive or not powered; two or three targets alone are descriptive. D-state is a separate preregistered primary hypothesis when Mode D is claimed.

**C9. G2a margin and freeze.** Non-inferiority delta is at most 0.05 absolute F1, chosen with an independent pilot set and measurement variance, frozen before unblinding or touching confirmatory test data. Report pilot SD and test SD separately. Never tune delta on test outcomes. Implement `freeze()` to write the hash and make test-evaluation code refuse to run without it.

**C10. F1 matching.** As defined in Section 8.6.

**C11. Retrospective labels and decoys.** Use the same endpoint (Ki with Ki, IC50 with IC50), the same assay type and compatible conditions. If assay uncertainty overlaps the fold-goal cutoff, label the pair indeterminate and exclude it from binary labels. AUC needs at least 100 actives and at least 100 matched decoys. EF at 1% needs at least 100 actives and at least 9,900 matched decoys. Report how decoys were generated and how well they match.

**C12. Pharmacophore-level leakage.** In addition to scaffold or ECFP4 Tanimoto >= 0.70 exclusion, exclude holo-derived labels whose same-type pharmacophore-set Tanimoto >= 0.80 or whose one-to-one feature-match F1 >= 0.80 within 1.0 angstrom of any held-out test label. Freeze these rules before splitting. The exclusion function is applied at Step 1 (seeds), Step 4 (training data) and Step 3 (calibration).

**C13. FEP replicates and budget.** Proposed shortlist 10 to 20 compounds (set in TCC and frozen). For m >= 1, test every off-target and the preregistered matched states. At least 3 independent replicas per alchemical leg or state. For m = 0, potency-only FEP against a same-target reference ligand or an absolute binding free energy, with no off-target leg. Report convergence and 95% confidence intervals. Confidence interval width <= 1.0 kcal/mol is a precision goal, not a guarantee. A confidence interval that crosses the threshold means unresolved (NOT\_EVALUABLE for that claim).

**C14. G4 primary contrast and claim boundary.** H5 primary contrast is Ensemble x PharmFlow versus Crystal x PharmFlow. Ensemble x DiffPhar versus Crystal x DiffPhar is secondary and explanatory. The report may say target-configurable; it may say universal only if adequately powered held-out-family and multi-family evidence supports it. Implement the claims list as a computed output of RunState, so the report cannot state a claim whose gate failed.

## 11. Step 6: Frozen GCPG molecule generation

**Purpose.** Turn pharmacophore models plus property settings into candidate molecules, written as SMILES strings, then place each molecule into the pocket by aligning it to the pharmacophore points.

**What GCPG is (from the CMD-GEN paper).** A Transformer encoder-decoder built on PGMG. The encoder reads a SMILES string and compresses it into a latent variable z (trained with a KL loss towards a standard normal prior). The decoder rebuilds the SMILES from z and a combined condition vector c\_com, which joins the pharmacophore condition and the gated property condition g(c\_feat). The pharmacophore is embedded with a graph attention network (EGAT; GT and GINE were tested and were worse). Loss terms: KL loss, language-model loss, and a mapping loss between heavy atoms and pharmacophore elements (using shortest-path distance on the molecular graph instead of Euclidean distance). Published settings: hidden size 384, 8 transformer blocks, Adam with learning rate 3e-4 and weight decay 1e-6, cyclic cosine annealing, 32 epochs, trained on more than 2 million ChEMBL 31 molecules after filtering (molecular weight above 800 removed, small fragments and metals removed, duplicates and invalid SMILES removed).

**In this project GCPG is frozen.** Do not retrain or fine-tune it. Use the released code and weights from the CMD-GEN repository. Build `adapters/gcpg.py` around them.

**Inputs.** Pharmacophore models from Step 5 (consensus, differential-filtered, optional state-specific); property gate values from the TCC (for example MW 400, LogP 3, QED 0.6, SAS 4; a channel set to null is closed, meaning unconstrained); the RunState (is confirmatory generation allowed?).

**Pre-check.** Read `RunState.confirmatory_generation_allowed`. If False (G4 failed or was NOT\_EVALUABLE), the run may generate only in exploratory mode from the crystal-structure baseline and every output is tagged `exploratory = True`. Never remove this tag later.

**Process.**

1. **Convert each PharmModel into the format GCPG expects.** GCPG is trained on pharmacophore hypotheses defined by feature types and pairwise distances, with topological (bond-path) distance in the mapping loss. The conversion from 3D pharmacophore points to GCPG's input format must follow the CMD-GEN repository exactly. Do not guess it: read the repository's generation script and copy the conversion. Unit-test it on one example from the paper's data.
2. **Set the property gates.** Pass the TCC values. Closed channels stay closed. Note that these are requests, not guarantees: the output clusters around the setting with spread (the paper's own plots show this).
3. **Sample.** For each hypothesis, draw latent z from a standard normal distribution and decode many SMILES (default 100 per hypothesis per repeat, 5 repeats). Fix seeds. Record hypothesis id, gate settings and seed for every molecule.
4. **Validate chemistry.** Parse each SMILES with RDKit, discard invalid ones (count them), canonicalise, remove duplicates.
5. **Generate 3D poses by alignment.** For each valid molecule, create RDKit conformers (several per molecule), then align each conformer to the pharmacophore points by finding the best rigid-body fit (Kabsch best-fit). Use the tolerance parameter: a value of 1 allows one unmatched pharmacophore point. Keep conformers that satisfy the tolerance; a molecule can have several valid poses. Compute the pharmacophore match score and record which points were unmatched.
6. **Record property achievement.** Compute MW, LogP, QED, SA score, TPSA and rotatable bonds with RDKit for every molecule and compare with the requested gates. Write a distribution plot and a table (requested vs achieved).

**Outputs.** `molecules.csv` (smiles, hypothesis id, seed, properties, match score, exploratory flag), `poses/` (aligned conformers), a generation manifest with counts at each stage (generated, valid, unique, aligned).

**Handoff.** Step 7 reads `molecules.csv` and poses.

**Common mistakes.**

- Retraining or fine-tuning GCPG. It is frozen.
- Inventing the conversion from 3D points to GCPG's input. Copy it from the repository.
- Treating property gates as hard limits and not checking achieved values.
- Over-constraining: tighter gating lowered the pharmacophore match ratio in the paper (78.25% to 74.48%). Pilot the gate values first, then freeze.
- Using a docking-score value as a gate and then also reporting docking as independent evidence. That is circular.
- Forgetting that molecules can exceed the pharmacophore count: the model may add structure beyond the sampled points. Do not assume every atom is pharmacophore-driven.
- Losing the exploratory tag.
- Generating from every state separately and pooling without recording the hypothesis id. Each molecule must remember its hypothesis.

**Acceptance tests.** (a) With G4 forced to FAIL, every output has `exploratory = True` and uses only the crystal baseline hypothesis. (b) Validity and uniqueness fractions are computed and logged. (c) For a known molecule and its own pharmacophore, alignment returns a match score of 1 (or the tolerance). (d) Requested versus achieved property table is produced.

## 12. Step 7: Early filters

**Purpose.** Remove molecules that are chemically invalid, reactive, unrealistic or too similar to known compounds, before spending compute on scoring. Apply identical filters to every arm and every target.

**Inputs.** `molecules.csv`, poses, exclusion regions from Step 5, reference ligands.

**Process (in this order, recording the count dropped at each).**

1. **Chemical validity.** RDKit sanitisation succeeds; one connected molecule; no unusual valence; remove salts and fragments by keeping the largest fragment only if that rule is declared in config.
2. **Structural alerts.** PAINS and Brenk filters (RDKit filter catalogs). Also flag reactive groups and polycyclic aromatic fused systems (the base paper notes these are a safety alert class). Decide in config whether a flag drops or only labels the molecule; apply the same rule everywhere.
3. **Novelty and scaffold split.** Compute Bemis-Murcko scaffolds and ECFP4 similarity against the training set and known actives. Report novelty. Keep a flag; dropping is a config choice.
4. **Synthesizability.** SA score threshold from config.
5. **Exclusion-volume clash filter.** If a differential model produced exclusion regions, reject poses whose heavy atoms fall in them. This is where exclusion volumes are applied, because GCPG cannot take them as input.
6. **Property sanity.** Verify requested property windows (informational unless config says to drop).
7. **Deduplicate** by canonical SMILES including stereochemistry rules declared in config.

**Outputs.** `filtered_molecules.csv`, an attrition table with counts after each filter.

**Handoff.** Step 8.

**Common mistakes.** Using different filters for different arms (this biases G4 and the 2x2 comparison). Filtering after docking. Dropping molecules silently without counts. Not canonicalising before deduplication. Treating stereo-variants as duplicates in one run and as different in another.

**Acceptance tests.** (a) A set of known PAINS molecules is removed. (b) The attrition table sums to the input count. (c) The same filter code path is used for both DiffPhar and PharmFlow arms (assert identical config hash).

## 13. Step 8: Scoring funnel \[FEP\]

**Purpose.** Rank the filtered molecules with increasingly expensive and increasingly accurate methods, so that only a short list reaches free-energy calculations and the lab. Docking is triage, not evidence of binding strength.

**Stages and typical sizes (illustrative; log the real counts).** About 10^5 generated, about 10^3 after filtering and docking, 20 to 50 after MM-GBSA rescoring, 10 to 20 for FEP and MD, 10 or fewer synthesised.

**Inputs.** Filtered molecules, the K retained states of the target and (if m >= 1) of each off-target, config (K, aggregation, shortlist size, selectivity goal).

### 13.1 Docking triage

1. Dock every molecule into each of the K target states. For m >= 1, dock into each off-target's K states. Use the same K and the same docking settings for every protein (AutoDock Vina with default parameters is the base paper's choice; the adapter hides the engine).
2. Convert scores to energy-like convention (more negative is better) at the adapter boundary.
3. Per protein, summarise each molecule: best-state score with the state id, median score, range, and K. Report ligand efficiency (score divided by heavy atom count) too, because larger molecules score better by size alone.
4. Rank molecules within each protein only. For triage selectivity use delta\_S\_dock = min over off-targets of \[S\_dock(O\_i) minus S\_dock(T)\], positive favouring T. This is for ordering only. Never use it as a hard cross-protein filter and never claim selectivity from it.
5. Apply the best-of-K null (C6) to any claim that uses best-state scores.
6. **Forbidden:** converting docking scores into Boltzmann weights or a free energy. Never write exp(-score/RT) over docking scores.

### 13.2 MM-GBSA rescoring (weak)

Rescore the top docking poses (per state) with MM-GBSA. Treat it as a weak rescoring filter with uncertainty, not as ground truth. Output a ranked list with per-state values.

### 13.3 FEP shortlist and gate

1. Select the shortlist (default 10 to 20, size frozen in TCC) from the MM-GBSA ranking using a declared rule (for example top by MM-GBSA with scaffold diversity). Freeze the rule before looking at FEP results.
2. Run free-energy calculations. For m >= 1, selectivity needs the binding free energy against the target and against every off-target, using absolute binding free energy or a thermodynamic cycle through a shared reference ligand with measured data on both proteins. Ordinary relative FEP only compares two ligands on the same protein and cannot by itself give cross-target selectivity.
3. At least 3 independent replicas per alchemical leg or state. Record convergence diagnostics and a 95% confidence interval for each result.
4. Define delta\_delta\_G = delta\_G\_bind(T) minus delta\_G\_bind(O). Negative favours T.
5. **Required threshold.** delta\_delta\_G <= minus RT ln(goal). At 298 K, RT = 0.593 kcal/mol, so a 10-fold goal needs about minus 1.36 kcal/mol and a 30-fold goal about minus 2.0 kcal/mol. Compute from the TCC goal; never hard-code. Convert back with fold = exp(minus delta\_delta\_G / RT) for reporting.
6. **State handling.** If state populations passed the shared convergence rule (C5), a population-weighted effective free energy may be used: delta\_G\_eff = minus RT ln of the sum over states of w\_k exp(minus delta\_G\_k / RT), with delta\_G\_k from FEP or MM-GBSA calibrated values, never raw docking. If not converged, use the worst-case state-specific delta\_delta\_G, which must itself meet the threshold, and its confidence interval must clear it.
7. **Outcome.** PASS only when the preregistered confidence interval clears the threshold. FAIL when the criterion is not met. NOT\_EVALUABLE when convergence or data are insufficient (including an interval that crosses the threshold). FAIL and NOT\_EVALUABLE both withdraw the FEP-supported affinity or selectivity claim. For m = 0: potency-only FEP against a same-target reference ligand or absolute binding free energy, no off-target leg.

**Outputs.** Per-protein docking tables, MM-GBSA table, FEP table with confidence intervals and convergence flags, the funnel attrition table, FEP GateResult.

**Handoff.** The final shortlist goes to Step 9.

**Common mistakes.**

- Boltzmann-weighting docking scores (forbidden).
- Comparing raw docking scores between proteins.
- Using relative FEP for cross-target selectivity.
- A single FEP replica, or ignoring the confidence interval when comparing with the threshold.
- Sign errors in delta\_delta\_G or delta\_S\_dock. Write unit tests on both conventions.
- Using different K, different docking settings or different box definitions for target and off-targets.
- Docking boxes that do not cover the equivalent pocket in each protein.
- Choosing the FEP shortlist after peeking at FEP outcomes.
- Treating a best-of-K score as if it were an unbiased single-state score.
- Mixing kcal/mol and kJ/mol.

**Acceptance tests.** (a) fold = exp(minus ddG / 0.593) gives about 10 at minus 1.36. (b) Sign test: a ligand that scores better on T than O gives positive delta\_S\_dock. (c) With non-converged weights, the code refuses to compute a weighted free energy. (d) A confidence interval that crosses the threshold returns NOT\_EVALUABLE. (e) For m = 0, no off-target docking or FEP call is made.

## 14. Step 9: MD and wet lab \[G7-8\]

**Purpose.** Check pose stability by simulation, then test the best compounds experimentally, and state which claims the data support.

**MD (code).** For each shortlisted compound and protein: at least 3 independent replicas; production length from config; measure ligand RMSD (stable if <= 2.0 angstrom) and key-contact persistence (default at least 70% of frames); report pose stability only. MD does not prove state transitions. State transitions need enhanced sampling or experiments. Write analysis scripts that read trajectories and output tables; do not hard-code file names.

**Wet lab (data ingestion, not code).** Wet-lab work cannot be coded. The software ingests assay results from a file with a fixed schema: compound id, protein, endpoint (IC50, Ki), value, relation (=, >, <), units, assay type, assay conditions, replicate id, validity flag (assay-valid or not). Test every off-target under matched conditions. Include biochemical potency, orthogonal binding where possible, cellular readouts and counter-screens as the TCC lists.

**Gate G7-8 (DOWNGRADE).**

- **Potency hit:** a synthesised, assay-valid compound that meets the TCC potency goal.
- **Selective hit:** additionally meets the fold goal against EVERY off-target, in the same endpoint and compatible assays.
- Hit rate = hits / synthesised, with a Wilson 95% confidence interval. If fewer than 5 compounds were synthesised, report descriptive results only. Comparative hit-rate claims need at least 30 synthesised compounds per arm and a prespecified power analysis. This is intentionally unavailable to small programmes. If the corresponding assay criterion fails, withdraw the experimental potency or selectivity claim, and report indeterminate labels separately.

**Common mistakes.** Treating a censored value such as IC50 greater than 10,000 nM as exactly 10,000 (a selectivity ratio computed from it is a bound, for example 'greater than 787-fold', never a point estimate). Mixing IC50 with Ki, or different assay formats. Counting compounds that were simplified or altered for synthesis as the same compound. Reporting only the best hits and omitting failures (report failures too, including non-selective compounds). Selecting compounds by eye without a recorded rule. Comparing against no baseline compounds when claiming superiority.

**Acceptance tests.** (a) Censored values propagate as bounds in the selectivity table. (b) A selective-hit call requires all off-targets present. (c) The Wilson interval for 1 of 5 is wide (roughly 0.4% to 64%; verify numerically) and the report labels it descriptive. (d) Mixing endpoints raises an error.

## 15. The pipeline driver

`run_pipeline.py` calls each step in order, passes only file paths, and updates `RunState` through `apply_consequence`. Pseudocode:

```
load tcc; validate; resolve_config; freeze_check()
rs = RunState()
r0 = step0(tcc);                    apply(rs, r0.G0)
if rs.target_run_stopped: report(rs); exit
step1(...)
r2 = step2(...);                    apply(rs, r2.Q)
if rs.ensemble_branch_stopped: rs.use_crystal_baseline_only = True
r3 = step3(...);                    apply(rs, r3.G1, r3.G2b)
r4 = step4(...);                    apply(rs, r4.G_LFO, r4.G2a)
r5 = step5(...);                    apply(rs, r5.G3, r5.D_state)   # D_state only in mode D
g4 = retrospective_G4(...);         apply(rs, g4)
step6(...)    # reads rs.confirmatory_generation_allowed; tags exploratory otherwise
step7(...); r8 = step8(...);        apply(rs, r8.FEP)
step9_ingest(...);                  apply(rs, r9.G7_8)
report(rs)    # lists claims allowed and withdrawn, every gate outcome
```

Rules for the driver: no step reads global variables; the report is generated only from the gate log and `RunState`, so it cannot state a claim whose gate failed; steps can be rerun from saved outputs.

## 16. Shared statistics utilities (write once, test thoroughly)

1. `grouped_paired_bootstrap(values_a, values_b, groups, n_boot, seed)` returns the difference and a confidence interval, resampling groups, not individual items.
2. `permutation_test(...)` with a supplied permutation scheme (within-block for G3, cluster permutation for G-LFO, compound-to-state permutation for best-of-K).
3. `holm_adjust(pvalues, alpha)` returns adjusted p-values and rejections; `holm_inverted_ci(...)`.
4. `wilson_interval(k, n, conf)`.
5. `hungarian_f1(generated, reference, tol, same_type_only)` (macro and micro).
6. `auc`, `enrichment_factor(scores, labels, fraction)`.
7. `best_of_k_null(score_matrix, K, n_perm, seed)`.
8. `weights_converged(replica_estimates)` (C5).
9. `chamfer_pocket_distance` with channel weights (C3).

Tests: known-answer cases (for example Wilson interval for 0 of 5; Holm on three p-values 0.01, 0.02, 0.04 at alpha 0.05), seed reproducibility, group bootstrap wider than item bootstrap on clustered data.

## 17. Testing the whole pipeline: a toy target in which every gate can fire

The gates are only a strength if they are shown running. Build a synthetic or small public toy target and a `--force` mode that degrades one input at a time. Required demonstrations (each saves a report):

1. Break 30% of conformers: Q = FAIL, ensemble branch stops.
2. Remove all reference states: G1 = NOT\_EVALUABLE, run goes unweighted.
3. Remove decoys: G4 = NOT\_EVALUABLE, confirmatory Step 6 blocked, outputs tagged exploratory.
4. Shrink families below 10 and 5: G-LFO = NOT\_EVALUABLE.
5. Swap labels randomly: G3 AUC near 0.5, DOWNGRADE.
6. Give a perfect pharmacophore model: G4 PASS.
7. Set off-targets to empty: all selectivity gates skipped, mode A path runs.

Also include: an end-to-end smoke test with mock adapters (mock aSAMt, mock PharmFlow, mock GCPG) that runs in minutes, and a determinism test (same seed, same config hash gives the same outputs).

## 18. Build order for the AI

1. Config schema, data classes, gate result and consequences, folder and manifest utilities.
2. Statistics utilities and their tests.
3. Leakage module (exclusion set, pharmacophore overlap, cluster splits) and pocket distance.
4. Step 0, Step 2, Step 3 with mock inputs.
5. Adapters with mocks (aSAMt, GCPG, docking, MM-GBSA, FEP, MD).
6. PharmFlow model, DiffPhar baseline, training and inference, F1 evaluation.
7. Step 5 consensus and differential, G3, D-state.
8. Retrospective scorer, G2b, G4.
9. Steps 6, 7, 8, 9 ingestion.
10. Driver, report generator, toy-target demonstrations.
11. Replace mocks with real adapters one at a time, rerunning the full test suite each time.

## 19. Decisions the AI must ask the user about, instead of guessing

1. How to call aSAMt (package, weights, sampling parameters).
2. GCPG repository version, weights, and the exact 3D-pharmacophore-to-input conversion.
3. Which docking program, MM-GBSA tool, FEP engine and MD engine are available.
4. Training data for PharmFlow (CrossDocked2020 or other) and where it is stored.
5. Pilot results for: vote fraction, grouping distance, GMM component rule, non-inferiority margin, property gate values, FEP shortlist rule, K.
6. Reference actives, inactives and decoys for the first target and the off-target list.
7. Compute budget (GPU count, hours).

## 20. Final checklist before accepting generated code

- No numeric literal for any threshold outside the config.
- Every gate returns PASS, FAIL or NOT\_EVALUABLE with a logged reason.
- NOT\_EVALUABLE is never treated as PASS.
- The leakage assertions run and are tested.
- Held-out data is not used in seeds, training, calibration, state definition or threshold choice.
- The same retrospective scorer serves G2b and G4.
- Docking is never exponentiated and never compared across proteins.
- FEP selectivity uses a cross-target-valid design and reports confidence intervals.
- The report lists claims strictly from the gate log.
- Seeds, versions and config hash are written for every run.
- Mock-adapter smoke test and the seven forced-failure demonstrations all pass.

*Known limits of this specification: the aSAMt and GCPG interfaces, and the PharmFlow design details (type handling, count head, equivariant backbone choice), are described at the level needed to build and must be confirmed against the real code and a pilot. PharmFlow has no existing implementation, so its results are unknown until trained and tested. Values marked as defaults come from earlier design versions or from the CMD-GEN paper and are frozen only after a pilot.*
