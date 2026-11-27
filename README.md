# DiffPhar: Equivariant Flow Matching for Pharmacophore Generation

Conditional pharmacophore generation in protein pockets using E(n)-equivariant Flow Matching.

## Structure
```
DiffPhar/
├── train.py                    # Training entry point
├── generate_phars.py           # Generate pharmacophores from PDB
├── test.py                     # Evaluate on test set
├── lightning_modules.py        # Main LightningModule
├── flow_matching.py            # Flow Matching implementation
├── dataset.py                  # Dataset loader
├── utils.py                    # Utilities
├── constants.py                # Constants & dataset params
├── process_crossdock.py        # Data preprocessing
├── configs/
│   ├── crossdocked_ca_cond.yml         # Local training config
│   ├── crossdocked_ca_cond_colab.yml   # Colab-optimized config
│   └── ... (other configs)
├── equivariant_diffusion/
│   ├── dynamics.py
│   ├── en_diffusion.py
│   ├── egnn_new.py
│   └── conditional_model.py
├── analysis/
│   ├── metrics.py
│   ├── visualization.py
│   ├── molecule_builder.py
│   └── docking.py
└── get_phar/                   # Pharmacophore extraction utilities
```

## Live run (ready to run)

Launch the front end from the project root with the `diffphar-fm` environment:

```powershell
& "C:\Users\hanut sai\miniconda3\envs\diffphar-fm\python.exe" -m streamlit run app.py
```

In the app, upload any complete protein–ligand complex from `sample_examples/complexes/`, keep the defaults (2 protein states, 6 molecules, Vina exhaustiveness 1), and click **Generate and dock molecules**. The app extracts the protein and calculates the pocket center from its bound ligand. Detailed run controls are inside **Advanced settings (optional)**. The real five-stage pipeline runs on the spot:

1. aSAM generates the conformational ensemble.
2. PharmFlow samples 3D pharmacophores.
3. Consensus GMM merges the states.
4. GCPG generates and embeds molecules.
5. AutoDock Vina docks every molecule across every state.

A verified live run with these defaults completed in about 110-130 seconds, produced 6/6 valid GCPG molecules, and passed Gates G0 and Q; the top candidate scored -8.72 kcal/mol mean. The app streams the full pipeline log while it runs. aSAM and PharmFlow use the GPU when available; aSAM falls back to CPU if CUDA fails. GCPG and Vina are CPU-only. For a larger run, use 2 states, 10 molecules, and exhaustiveness 2; a verified run of that size produced 10/10 valid GCPG molecules and passed G0/Q, with docking dominating the runtime. All other gates remain **NOT EVALUATED** because no independent benchmark or experimental data are included.

### Environments used

- `diffphar-fm` (`C:\Users\hanut sai\miniconda3\envs\diffphar-fm\python.exe`): app, aSAM (CUDA when available, CPU fallback), PharmFlow (CUDA), pipeline orchestration.
- `gcpg` (`C:\Users\hanut sai\miniconda3\envs\gcpg\python.exe`): GCPG generation (DGL, CPU-only environment).
- `diff_phar` (`C:\Users\hanut sai\miniconda3\envs\diff_phar\python.exe`): Meeko 0.8 with RDKit 2026.03.6. Do not use the `diffphar-fm` environment for Meeko; its RDKit 2022.09.5 is incompatible with Meeko 0.8.
- `vina.exe` in the project root: AutoDock Vina 1.2.7 (CPU-only).
- GPU: 4 GB VRAM is sufficient; aSAM uses about 1.2 GB and PharmFlow under 1 GB.
- PharmFlow checkpoint: `checkpoints_pharmflow/best_model_epoch_46.ckpt`.
- GCPG weights and tokenizer: `checkpoints_gcpg/model_checkpoints/checkpoints/`.

The CLI resolves the project checkpoint, GCPG Python, Meeko Python, and Vina automatically:

```powershell
& "C:\Users\hanut sai\miniconda3\envs\diffphar-fm\python.exe" run_full_pipeline.py `
  --pdb sample_examples\receptors\1FKB_receptor.pdb `
  --pocket_center 3.109 12.167 13.353 --pocket_radius 10 `
  --k_states 2 --n_mol 6 --exhaustiveness 1 `
  --out_dir pipeline_run_output --device cuda --allow_bad_receptor_residues
```

### Known limitations

- The 1M17 example can fail Gate Q in multi-state mode because the sampled states differ by more than the 2 Å C-alpha RMSD threshold. The pipeline now resamples and falls back to 1 state automatically if needed; use 1FKB for a true multi-state run.
- Vina scores are docking rankings, not free energies or potency measurements. The FEP and G7-8 gates are intentionally not evaluated.
- `train.py` needs the preprocessed dataset (`data_raw/processed_crossdock_noH_ca_only_temp/`), which is not included in this checkout. Training is not part of the app run.
- `test.py` needs a labeled test directory; it is not needed for the app run.

## Quick Start (Local)

### Try the front end with sample protein pockets

Install the front-end dependency (`pip install streamlit`) and launch `streamlit run app.py`. Upload any full `.pdb` or `.cif` complex from [sample_examples/complexes](sample_examples/complexes/). The app identifies the bound ligand, extracts the protein receptor, and calculates the pocket center automatically. See [sample_examples/README.md](sample_examples/README.md) for the list of examples and sources. The five pipeline stages still require the model checkpoints and executables configured in the app.

```bash
# Install dependencies
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Preprocess data (CA-only)
python process_crossdock.py data_raw/ --ca_only --dist_cutoff 8.0

# Train
python train.py --config configs/crossdocked_ca_cond.yml

# Generate pharmacophores with the bundled full-atom checkpoint
python generate_phars.py checkpoints_pharmflow/best_model_epoch_46.ckpt \
  --pdbfile sample_examples/receptors/1FKB_receptor.pdb \
  --resi_list A:54 A:55 A:56 A:48 A:53 A:46 A:47 A:87 A:59 A:49 A:82 A:81 A:57 A:52 A:37 \
  --n_samples 20 --num_nodes_phar 10
```

## Colab Setup (Pro+ / Standard)

### Step 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

> **Important (For Shared Folders):**  
> If the dataset folder (`DRUG DISVOVERY DATASETS` or `G1591 DATASET`) was shared with your Google account, Google Colab cannot access it under `/content/drive/MyDrive/` directly from "Shared with me".  
> **To fix this:**
> 1. Open [Google Drive](https://drive.google.com).
> 2. Go to **"Shared with me"**.
> 3. Right-click the folder (`DRUG DISVOVERY DATASETS` or `G1591 DATASET`).
> 4. Select **Organize** > **Add shortcut** > Select **"My Drive"** > Click **Add**.  
> This maps it directly to `/content/drive/MyDrive/DRUG DISVOVERY DATASETS/G1591 DATASET/`.

---

### Step 2: Clone Repo & Install Dependencies
```bash
!git clone -b diffphar-dev https://github.com/amog656/CMD-fork.git
%cd CMD-fork
!pip install -r requirements.txt
```

---

### Step 3: Launch Training

* **Full-Atom Training (Recommended, 20 epochs):**
```bash
!python train.py --config configs/crossdocked_full_cond_colab.yml
```

* **$C_\alpha$-Only Training (Lightweight / T4 GPU):**
```bash
!python train.py --config configs/crossdocked_ca_cond_colab.yml
```

---

### Step 4: Auto-Resume if Training Stops / Disconnects

The training script automatically creates checkpoints at the end of **every epoch** and saves `last.ckpt` under:
`/content/drive/MyDrive/DRUG DISVOVERY DATASETS/G1591 DATASET/outputs/<run_name>/checkpoints/`

**If training disconnects or stops:**
Simply reconnect and re-run the same command:
```bash
!python train.py --config configs/crossdocked_full_cond_colab.yml
```
`train.py` automatically detects `last.ckpt`, prints:
`==> [Auto-Resume] Found existing last checkpoint at: .../last.ckpt`
and resumes model weights, optimizer momentum, learning rates, and epochs seamlessly from where it stopped.

*(Optional)* You can also explicitly specify a checkpoint to resume from:
```bash
!python train.py --config configs/crossdocked_full_cond_colab.yml \
  --resume "/content/drive/MyDrive/DRUG DISVOVERY DATASETS/G1591 DATASET/outputs/crossdocked_full_cond_colab/checkpoints/last.ckpt"
```

## Config Differences

| Setting | Local (`crossdocked_ca_cond.yml`) | Colab (`crossdocked_ca_cond_colab.yml`) |
|---------|-----------------------------------|----------------------------------------|
| `batch_size` | 4 | 8 |
| `num_workers` | 0 | 4 |
| `n_epochs` | 1000 | 100 (test) |
| `precision` | 32 | 16-mixed (via Trainer) |
| `datadir` | Local path | `/content/drive/MyDrive/DiffPhar_data/...` |
| `logdir` | Local path | `/content/drive/MyDrive/DiffPhar_results/...` |

## Generation

```bash
# From PDB with residue list
python generate_phars.py CHECKPOINT.ckpt \
  --pdbfile POCKET.pdb \
  --resi_list A:24 A:25 A:26 \
  --n_samples 50 --num_nodes_phar 10

# From PDB with reference ligand
python generate_phars.py CHECKPOINT.ckpt \
  --pdbfile POCKET.pdb \
  --ref_ligand A:1101 \
  --n_samples 50
```

## Evaluation

```bash
python test.py CHECKPOINT.ckpt \
  --test_dir data_raw/processed_crossdock_noH_ca_only_temp/test/ \
  --sanitize
```

## Integrated pipeline

`run_full_pipeline.py` is the inference orchestrator. It requires a matching PharmFlow checkpoint (`--ckpt`), the GCPG checkpoint and tokenizer under `checkpoints_gcpg/model_checkpoints/checkpoints/`, an input PDB accepted by aSAM, and locally installed aSAM weights. The bundled aSAM inference implementation is Linux-oriented; install it and its dependencies separately with `pip install -e asam_engine` on a supported Linux environment. The integrated runner uses the bundled Atlas aSAM model on CPU and will stop if its weights are unavailable; it does not synthesize coordinate perturbations as a substitute.

The Python environment running the orchestrator needs the packages in `requirements.txt` and aSAM's dependencies. GCPG uses its compatible PyTorch/DGL/fairseq environment. Meeko and a compatible RDKit must be installed together in the environment selected by `--meeko_python` (defaults to `--gcpg_python`); its `mk_prepare_ligand` and `mk_prepare_receptor` commands prepare docking inputs. Example:

```bash
python run_full_pipeline.py \
  --pdb asam_engine/examples/input/6h49_A.pdb \
  --pocket_center X Y Z \
  --ckpt /path/to/pharmflow.ckpt \
  --gcpg_python /path/to/python-with-gcpg-dependencies \
  --meeko_python /path/to/python-with-meeko-and-rdkit \
  --n_mol 15 --device cpu
```

Replace `X Y Z` with the binding-site center measured in the receptor PDB coordinate frame. This input is required because the example receptor contains no bound ligand from which to infer the pocket. PharmFlow selects nearby atoms/residues within `--pocket_radius` (default 10 Å).

The generated report separates checks that ran from gates marked `NOT EVALUATED`. Comparative performance gates require independent benchmark datasets; selectivity requires an off-target calculation; potency requires measured dose-response data. Vina docking scores are computational ranking signals and do not establish binding affinity, IC50, safety, or efficacy. No retrospective or experimental claims should be inferred from a pipeline run alone.

## Front end

The Streamlit app accepts a complete protein-ligand `.pdb` or `.cif` complex. It extracts a protein chain and finds the pocket center from the bound ligand, so no pocket coordinates are needed. Choose model environments and run size only if you want to change the defaults. Start it from the PharmFlow environment:

```powershell
& 'C:\Users\hanut sai\miniconda3\envs\diffphar-fm\python.exe' -m streamlit run app.py
```

The app writes each run under `pipeline_runs/`. Enter the pocket center in the same PDB coordinate frame; the zero values shown initially are placeholders, not a recommended center. Keep Meeko's unmatched-residue option off unless you have inspected and accepted any residues it may omit.

## Model

- **Architecture**: E(n)-equivariant Graph Neural Network (EGNN)
- **Framework**: Flow Matching (conditional on protein pocket)
- **Representation**: CA-only or full-atom pocket
- **Output**: 3D pharmacophore coordinates + types (8 classes)

## Pharmacophore Types

1. Aromatic
2. Hydrophobe
3. PosIonizable
4. NegIonizable
5. Acceptor
6. Donor
7. LumpedHydrophobe
8. Others

## Data

CrossDocked2020 dataset (pocket10 split). Download separately and place in `data_raw/crossdocked_pocket10/`.

## License

Research use only.
