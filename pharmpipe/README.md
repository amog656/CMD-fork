# Pharmpipe

Mock-capable implementation scaffold for the Ensemble-Aware Pharmacophore Pipeline.
The TCC schema, run-state/gate handling, shared statistics, leakage utilities, file
stages, adapters, and a toy run live in this package. A CMD-fork DiffPhar Flow
Matching inference adapter is available for PharmFlow; the other external tools
remain mocks or unavailable.

## Run the toy target

From the project root:

```powershell
python -m pharmpipe.run_pipeline --tcc pharmpipe/toy_data/toy_tcc.yaml --output-root pharmpipe/runs
```

The toy config uses synthetic settings and data. The run is expected to remain
exploratory: full physical ensemble QC, trained PharmFlow, DiffPhar, and experimental
data are unavailable, so their dependent gates return `NOT_EVALUABLE`.

## Use a precomputed aSAM ensemble

Step 1 can ingest an already generated ensemble instead of calling the mock sampler.
Set these values in the TCC `thresholds` block, keeping the count equal to the number
of frames in the input folder:

```yaml
thresholds:
  n_samples_per_seed: 20
  precomputed_ensemble_dir: /absolute/path/to/asam_ensembles_unique/1bv7_A_rec
```

The directory must contain `conf_*.pdb` files and a `.done` marker whose contents are
`COMPLETED_20_FRAMES` (with the number matching the configured count). Every frame
must have the same atom identities in the same order. The run copies the PDBs into
its own `step1` folder and records the source directory in the manifest. It also
hashes the marker and input frames when creating the run identity. This only connects
the aSAM output to Step 1; later physical QC and modeling stages still use the
project's existing mock or unavailable adapters.

Generate the seven gate demonstrations:

```powershell
python -m pharmpipe.toy_target --output-root pharmpipe/demos --force all
```

Use `--force` with one of `q_break_30`, `no_reference_states`, `no_decoys`,
`small_families`, `shuffled_labels`, `perfect_pharmacophore`, or `mode_a` to run a
single demonstration. Existing run and demo folders are never overwritten.

## Tests

```powershell
python -m pytest pharmpipe/tests -q
```

The suite covers TCC validation and freezing, deterministic mocks, gates and
consequences, the statistics and leakage modules, the pharmacophore graph distances,
the end-to-end mock run, and all forced gate demonstrations.

## Scientific scope

The pipeline reports `NOT_EVALUABLE` wherever the mock build lacks evidence. In
particular, Step 2 checks broad backbone bond geometry and chain continuity, writes
an advisory heavy-atom clash screen, and computes aligned C-alpha RMSD only when it
can match at least three pocket residues to the reference structure. It does not
evaluate full stereochemistry or energies and does not perform restrained relaxation,
so Q remains `NOT_EVALUABLE` without those pieces. PharmFlow generalization and
comparison are withdrawn while the model is untrained. G4 is `NOT_EVALUABLE` in
ordinary runs because confirmatory PharmFlow/DiffPhar arm evidence is unavailable.
These safeguards keep mock outputs from being presented as experimental or
retrospective validation.

Before real adapters are connected, the TCC must supply every threshold used by a
stage and the real repository/tool interfaces must be confirmed. The GCPG graph
converter currently permits only an explicitly requested Euclidean mock encoding;
it refuses to guess the repository's input encoding.

## Connect a CMD-fork PharmFlow checkpoint

To use the early Flow Matching checkpoint for Step 4, add these settings to the
TCC `thresholds` block. Set the repository directory to a local checkout of the
`diffphar-dev` branch and use the Python executable from the environment where
that checkout's dependencies are installed:

```yaml
thresholds:
  pharmflow_repository_dir: /absolute/path/to/CMD-fork
  pharmflow_checkpoint_path: /absolute/path/to/best-model-epoch=epoch=01.ckpt
  pharmflow_python_executable: python
  pharmflow_samples_per_state: 5
  pharmflow_timesteps: 15
```

The adapter runs inference in a worker process, pools generated points by feature
type, and selects Gaussian mixture components by BIC to make one pharmacophore per
pocket state. The checkpoint and repository Python files are included in the run
identity. This is exploratory integration: this project currently has no training
pocket reference set/OOD calibration wired into Step 3, so its in-domain filter
does not admit states by default. It also does not establish model quality or
validation; the supplied checkpoint is an epoch-one checkpoint. Keep the real
domain filter enabled and treat G-LFO/G2a as `NOT_EVALUABLE` until proper held-out
family and matched-baseline evaluations are available.
