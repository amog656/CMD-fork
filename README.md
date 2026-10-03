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

## Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Preprocess data (CA-only)
python process_crossdock.py data_raw/ --ca_only --dist_cutoff 8.0

# Train
python train.py --config configs/crossdocked_ca_cond.yml

# Generate pharmacophores
python generate_phars.py checkpoints/epoch=XX.ckpt \
  --pdbfile data_raw/processed_crossdock_noH_ca_only_temp/test/1phk-A-rec-1phk-atp-lig-tt-min-0-pocket10.pdb \
  --resi_list A:24 A:25 A:26 A:27 A:28 \
  --n_samples 20 --num_nodes_phar 10
```

## Colab Setup

1. **Clone repo:**
```bash
git clone -b diffphar-dev https://github.com/YOUR_USERNAME/CMD-GEN.git
cd CMD-GEN/DiffPhar
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

2. **Mount Google Drive (for data & checkpoints):**
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. **Upload processed data to Drive:**
   - Local: `data_raw/processed_crossdock_noH_ca_only_temp/` → Drive: `/MyDrive/DiffPhar_data/`

4. **Train on Colab:**
```bash
python train.py --config configs/crossdocked_ca_cond_colab.yml \
  --resume /content/drive/MyDrive/DiffPhar_checkpoints/last.ckpt  # optional
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