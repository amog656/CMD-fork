# CMD-GEN vs. the Current Project Model

**Results report — 26 November 2026**

## Scope and conclusion

This report compares the published CMD-GEN method with the current local implementation in this project. The latter is best described as a **CMD-GEN-derived Flow Matching pipeline**: it replaces the published DiffPhar diffusion sampler with the project's PharmFlow conditional Flow Matching sampler, adds aSAM-generated protein conformations, and keeps the consensus GMM and a GCPG molecule generator. It is therefore a component-level extension of CMD-GEN, not an independently developed end-to-end model. The published CMD-GEN architecture uses pocket-conditioned pharmacophore diffusion, GCPG, and pharmacophore-based conformational alignment. [Zou et al., *Communications Biology* (2025)](https://doi.org/10.1038/s42003-025-07840-3); [CMD-GEN source repository](https://github.com/zyrlia1018/CMD-GEN).

**The current workspace still does not establish that the Flow Matching pipeline outperforms CMD-GEN.** A small matched sampler ablation and common-downstream pilot has now been run on three CMD-GEN case-study targets plus one extracted held-out subset case (AROE_THET8). It yields lower ligand-feature recovery for PharmFlow on all four targets, while end-to-end docking is mixed: mean Vina favors DiffPhar on three of four targets and PharmFlow on 7NI4. PharmFlow is roughly 12x faster and produces higher-QED molecules closer to the requested MW on three of four targets. This is not an overall performance win for either model. The experiment uses one seed, a small target set, and no verified non-overlapping held-out split, so the results are exploratory. The official release contains a 100-case test-name manifest; one of five extracted matched protein–ligand examples has been benchmarked and four remain. There is no selectivity or wet-lab data for the current model.

## Model and method comparison

| Component | Published CMD-GEN | Current project implementation |
|---|---|---|
| Pocket-conditioned pharmacophore sampler | DiffPhar, an equivariant diffusion model; the paper reports 500 denoising steps and separate full-atom and Cα representations. | PharmFlow, conditional Flow Matching. The local checkpoint metadata identify `crossdock_full`, full-atom representation, epoch 46 (checkpoint epoch field 45), 1,123,920 optimizer steps; the runtime inferred 11 pocket atom types, 8 pharmacophore types, hidden size 256, 4 EGNN layers. Sampling in the pipeline uses 50 Euler steps. |
| Protein flexibility | The paper's core benchmark describes static protein pockets. | aSAM generates conformational states. The app defaults to 2 states; the current quick pilot runs below deliberately used 1 state. Gate Q may trigger retries or reduce the requested state count. |
| Pharmacophore consensus | GMM-based clustering is part of the published framework. | GMM clusters sampled features across the generated states and emits a consensus `.posp` blueprint. |
| Molecule generation | GCPG generates property-conditioned SMILES from pharmacophore graphs. | GCPG is retained. The local runner loads a 384-hidden, 8-layer GCPG model and conditions on MW 400, logP 3.5, QED 0.65, SAS 3, and 5 rotatable bonds. Exact weight identity with the published checkpoint has not been verified. |
| 3D conformation and docking | CMD-GEN aligns generated conformations to sampled pharmacophores; the paper evaluates generated-versus-docked conformation RMSD. | The local code embeds generated SMILES with RDKit and UFF, then prepares and docks them with Meeko/Vina. The local reports do not calculate ligand pose RMSD against a crystal reference. |
| Selectivity | CMD-GEN compares aligned on-target/off-target pharmacophore clouds and reports PARP1/2 experimental results. | The current pipeline's selectivity, FEP, and experimental gates are marked **NOT EVALUATED**. The local sample runs do not include an off-target structure or wet-lab measurements. |

The current checkpoint is not directly matched to the published training protocol: the paper reports a sequence-identity-controlled CrossDocked test setup (about 100,000 training complexes and 100 held-out pockets), while the official release contains a 100-case test-name manifest, but PharmFlow training overlap with it is unknown. The current checkpoint's training path is CrossDocked, but its original data and split manifest are absent. The matched pilot below controls the pocket atoms and downstream GMM/GCPG/docking stages on three selected structures plus one extracted held-out subset case (AROE_THET8); it does not verify target non-overlap with PharmFlow's training data or isolate training-data and training-protocol differences. Four of the five extracted held-out cases remain unrun.

## Results reported for published CMD-GEN

### GCPG molecule-generation benchmark

The paper's GCPG-EGAT row reports the following benchmark values. These are **published module-level results**, not results for the entire current pipeline.

| Published method | Validity | Uniqueness | Novelty | Available-molecule ratio |
|---|---:|---:|---:|---:|
| GCPG-EGAT | 0.975 | 0.998 | 0.983 | 95.6% |

The evaluation used the paper's ChEMBL/GuacaMol generation setup and metrics. The separate app-validation pilot described below has only four generated molecules across two different proteins, so its observed 4/4 valid and unique molecules are descriptive and are **not statistically comparable** to this benchmark. Novelty against the training corpus and an available-molecule ratio have not been calculated for the local molecules. [CMD-GEN paper, Table 1 and Methods](https://doi.org/10.1038/s42003-025-07840-3).

### Target-specific docking and experimental results

For its selected top-scoring molecules, the paper reports CMD-GEN(R1)/CMD-GEN(R2) Vina scores of −11.89/−10.43 kcal/mol for ATM, −13.58/−12.26 for PARP1, and −13.32/−11.82 for USP1. These are best-molecule examples from the paper's target-specific generation experiment, not cross-target means. The paper also reports that synthesized optimization-mode compound Y5 had a PARP1 IC50 of 12.7 nM and a selectivity ratio above 787-fold for PARP1 over PARP2. [CMD-GEN paper, Figs. 5 and Table 2](https://doi.org/10.1038/s42003-025-07840-3).

The paper further reports that more than 75% of CMD-GEN(R2) conformations were within 2 Å RMSD of docked conformations across the three example targets. The current local pipeline has not measured the same RMSD endpoint. [CMD-GEN paper, Fig. 7](https://doi.org/10.1038/s42003-025-07840-3).

## Current project pilot results

The following fresh local runs used the downloaded RCSB structures, one aSAM state, two molecules per target, and Vina exhaustiveness 1. Each run passed G0 and Q. The run reports explicitly leave comparative benchmarks, selectivity, FEP, and experimental gates unevaluated.

| Target example | Molecules generated / valid | Individual Vina scores (kcal/mol) | Mean MW | Mean QED | Mean logP | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| FKBP12–rapamycin, PDB 1FKB | 2 / 2 | −7.64, −6.67 | 399.6 | 0.597 | 3.95 | 47.26 s |
| EGFR–erlotinib, PDB 1M17 | 2 / 2 | −7.29, −7.16 | 471.7 | 0.593 | 4.22 | 72.01 s |

Across these four molecules, RDKit parsing found four valid and four distinct canonical SMILES. The 1M17 pilot's generated molecules averaged about 472 Da despite an MW conditioning target of 400 Da; this illustrates that the GCPG condition is not an exact property guarantee. The two targets, sample size, receptor structures, one-state protocol, and Vina settings differ from the published CMD-GEN target-specific benchmark. **These scores must not be subtracted from the published ATM/PARP1/USP1 scores or used to claim better or worse binding.**

Run evidence: [1FKB report](pipeline_runs/manual_test_20261126_1532/PIPELINE_VALIDATION_REPORT.md), [1FKB SDF](pipeline_runs/manual_test_20261126_1532/step4_molecules/consensus_blueprint_candidates_3d.sdf), [1M17 report](pipeline_runs/internet_1M17_test_20261126/PIPELINE_VALIDATION_REPORT.md), [1M17 SDF](pipeline_runs/internet_1M17_test_20261126/step4_molecules/consensus_blueprint_candidates_3d.sdf).

## Interpretation for a manuscript

The supportable finding at this stage is an **exploratory matched ablation**: PharmFlow is substantially faster than DiffPhar on the tested pockets, but has lower ligand-feature recovery on all four. After passing both samplers through the same local GMM/GCPG/RDKit/Vina stages, target-level mean docking favors PharmFlow for ATM (7NI4) and DiffPhar for PARP1 (7ONS), USP1 (8A9K), and the held-out AROE_THET8 case. Across the four target means, mean Vina is −8.86 kcal/mol for DiffPhar and −8.01 for PharmFlow. PharmFlow has higher QED on three of four targets. This is mixed pilot evidence, not an overall performance win; it says nothing about selectivity or biological activity.

### Suggested manuscript wording

> We compared the official full-atom CMD-GEN DiffPhar checkpoint with the PharmFlow checkpoint using matched pocket atoms and 50 eight-point samples per sampler on three CMD-GEN case-study structures (PDB 7NI4, 7ONS, and 8A9K) plus 10 samples per sampler on one extracted held-out case (AROE_THET8; one seed). DiffPhar achieved higher bound-ligand pharmacophore-feature F1 on all four targets, while PharmFlow sampling was approximately 12-fold faster. In a shared-downstream pilot using the same local GMM, GCPG, RDKit/UFF, and Vina protocol, mean docking scores favored PharmFlow for 7NI4 and DiffPhar for 7ONS, 8A9K, and AROE_THET8. The four-target, one-seed study is exploratory, uses the current project's GCPG rather than the published CMD-GEN GCPG checkpoint, and does not establish overall superiority or biological activity.

Use this wording only for the exploratory pilot. A controlled held-out benchmark with repeated seeds, uncertainty estimates, and native end-to-end baselines is still needed for a superiority claim.

## Minimum experiment needed for a defensible head-to-head claim

1. **Done for the sampler pilot:** obtain and verify the official DiffPhar checkpoint/code, and record its hash. Local PharmFlow and shared GCPG checkpoint hashes are recorded in the benchmark manifest. The official CMD-GEN GCPG weights are not the downstream generator used in the controlled pilot.
2. Run both systems on the **same held-out protein–ligand complexes**, with the same receptor preparation, pocket center, ligand references, property conditions, number of generations, and random-seed schedule. Prefer the published 100-pocket test split if its data and identities can be recovered; otherwise define and publish a non-overlapping test set.
3. **Sampler-only attribution is piloted:** GMM/GCPG and the downstream docking setup were shared for the three-target run and one held-out case. Repeat on the full verified held-out pocket set, and separately run each method's native downstream pipeline if making a whole-method claim.
4. Increase to hundreds of generations per condition across repeated seeds and targets. Report validity, uniqueness, novelty against the same training set, available-molecule ratio, QED, MW, logP, SAS, rotatable bonds, pharmacophore match, Vina score distributions, and ligand efficiency. Include confidence intervals and target-level results, not only the best molecule.
5. Compare generated conformations with the same crystal or reference poses using symmetry-corrected RMSD and a stated threshold such as 2 Å. The current RDKit-plus-Vina path and CMD-GEN PharAlign path are different pose-generation procedures, so this endpoint must be evaluated explicitly.
6. If claiming selectivity, use matched on-target/off-target pairs and the same docking protocol, then validate prioritized compounds experimentally. The current example runs provide no selectivity evidence.

## Direct sampler and common-downstream pilot

The official [CMD-GEN Zenodo release](https://zenodo.org/records/13841142) was downloaded and passed its published MD5 check (`77a8959db7944bc8a68fb865d34a8b5e`). Its full-atom DiffPhar checkpoint SHA-256 is `4AC63D9F60FC86779DFE4572A77D87F500EB0E7AFFE9AEACB5AEF0CA80407755`. The local PharmFlow and GCPG checkpoint SHA-256 values are recorded in [`benchmark_protocol.json`](benchmark_protocol.json).

I ran 50 samples per sampler on each of the three CMD-GEN case-study pockets (ATM 7NI4, PARP1 7ONS, USP1 8A9K), with seed 11 and eight pharmacophore points per sample. I also ran the first extracted held-out subset case, AROE_THET8, with 10 samples per sampler under the same settings. Both models received the same ligand-centered pocket atoms. DiffPhar used 500 denoising steps; PharmFlow used 50 flow steps. I scored one-to-one feature-family matches to RDKit features from each bound ligand, using a 2 Å distance threshold. All generated point clouds were finite. These are selected case studies plus one held-out example, not the full 100-pocket held-out test set, and they do not establish generalization.

| Target | DiffPhar feature F1 | PharmFlow feature F1 | DiffPhar runtime (s) | PharmFlow runtime (s) |
|---|---:|---:|---:|---:|
| ATM (7NI4) | 0.336 | 0.205 | 239.9 | 19.3 |
| PARP1 (7ONS) | 0.089 | 0.052 | 253.5 | 20.0 |
| USP1 (8A9K) | 0.177 | 0.121 | 241.7 | 19.2 |
| AROE_THET8 (held-out subset; 10 samples) | 0.368 | 0.160 | 98.2 | 8.4 |

DiffPhar had higher ligand-feature F1 on all four pockets in this pilot. PharmFlow was about 12.6 times faster on the 50-sample pilot and 11.6 times faster on the 10-sample held-out case; its point clouds were more diverse under the type-aware cloud-distance measure. The one-seed, four-target setup has no statistical uncertainty estimate; the diversity value is not itself a quality metric. The held-out PharmFlow run metadata recorded a Downloads copy of the checkpoint; its SHA-256 matches the project checkpoint recorded in `benchmark_protocol.json`.

To test whether sampler differences carry through molecule generation, I passed both sets through the **same local** GMM, GCPG checkpoint/tokenizer, MW/logP/QED conditions, RDKit/UFF conformer builder, receptor preparation, and Vina v1.2.7. Each arm requested 20 molecules per target; Vina used exhaustiveness 1 and the same ligand-centered 22 Å box for both arms. The held-out AROE_THET8 check used 10 sampler clouds per arm instead of 50, but the same downstream settings. This is a controlled sampler ablation with shared downstream components, not a reproduction of vanilla CMD-GEN's native GCPG checkpoint and PharAlign stage.

| Target | Arm | Valid unique / 20 | Mean MW | Mean QED | Mean logP | Mean Vina (kcal/mol) | Median | Best |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ATM (7NI4) | DiffPhar | 19 | 493.0 | 0.355 | 5.30 | −7.788 | −8.102 | −9.124 |
| ATM (7NI4) | PharmFlow | 19 | 511.6 | 0.339 | 5.38 | **−8.537** | −8.658 | **−10.500** |
| PARP1 (7ONS) | DiffPhar | 19 | 448.3 | 0.500 | 3.61 | **−10.082** | **−10.490** | **−11.510** |
| PARP1 (7ONS) | PharmFlow | 17 | 410.7 | **0.630** | 3.83 | −7.620 | −7.153 | −10.120 |
| USP1 (8A9K) | DiffPhar | 17 | 561.8 | 0.366 | 6.03 | **−9.250** | **−9.517** | **−10.690** |
| USP1 (8A9K) | PharmFlow | 19 | 412.9 | **0.604** | 4.32 | −8.718 | −8.973 | −9.947 |
| AROE_THET8 (held-out; 10 samples) | DiffPhar | 20 | 449.5 | 0.499 | 3.63 | **−8.321** | **−8.311** | **−9.535** |
| AROE_THET8 (held-out; 10 samples) | PharmFlow | 16 | 424.2 | **0.632** | 4.02 | −7.161 | −7.072 | −8.076 |

The target-level mean Vina score favored PharmFlow on ATM (7NI4) and DiffPhar on PARP1 (7ONS), USP1 (8A9K), and held-out AROE_THET8. Across the four target means, mean Vina was −8.86 kcal/mol for DiffPhar versus −8.01 for PharmFlow; the paired difference (PharmFlow minus DiffPhar) was −0.75, +2.46, +0.53, and +1.16 kcal/mol, respectively (positive means DiffPhar scored better). PharmFlow had higher QED and was closer to the requested MW of 400 Da on three of four targets; those are generated-molecule properties, not evidence of stronger binding. This pilot therefore gives **mixed end-to-end docking results**, while the direct sampler feature-recovery measure favored DiffPhar on all four targets. It does not show that the new model is better overall.

Vina scores are approximate ranking signals, not measured binding affinities, selectivity, safety, or drug efficacy. The pilot uses one seed, three selected case studies plus one extracted held-out case, 10 held-out sampler clouds per arm versus 50 for the case studies, 20 requested molecules per arm/target, and exhaustiveness 1. It has no confidence intervals or statistical significance tests. The official 100-case test-name manifest has been recovered, but PharmFlow training overlap is unknown; one of the five extracted held-out cases has been benchmarked and four remain. The full settings, checkpoint hashes, and limitations are in [`benchmark_protocol.json`](benchmark_protocol.json); raw outputs and per-sample scores are in `results/cmdgen_sampler_pilot/`, `results/cmdgen_end_to_end_pilot/`, and `results/heldout_comparison/`.

## References

1. Zou Y, Guo T, Fu Z, et al. **A structure-based framework for selective inhibitor design and optimization.** *Communications Biology*. 2025;8:422. [https://doi.org/10.1038/s42003-025-07840-3](https://doi.org/10.1038/s42003-025-07840-3).
2. Zou and colleagues. **CMD-GEN source repository.** [https://github.com/zyrlia1018/CMD-GEN](https://github.com/zyrlia1018/CMD-GEN).

