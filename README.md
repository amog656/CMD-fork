# GANDIVA: Target-Configurable Ensemble-Aware Pharmacophore Pipeline

GANDIVA is an end-to-end structure-based computational drug design platform. It integrates continuous Flow-Matching (`PharmFlow`), frozen graph-conditioned molecule generation (`GCPG`), physical biophysical docking (`AutoDock Vina`), and an interactive WebGL discovery interface.

---

## Quick Start (For Windows)

### 1. Requirements
Ensure you have Python 3.11 installed. Install required packages:
```bash
pip install torch rdkit dgl fairseq joblib scipy tqdm
```

### 2. Download Model Weights
Because the trained GCPG neural network weights (`final_model.pth`, 679 MB) exceed GitHub's 100MB file limit, download them once using our included script:
```bash
python download_weights.py
```
*(Alternatively, download `final_model.pth` directly from [Zenodo 13841142](https://zenodo.org/records/13841142) and place it in `cmdgen_repo/GCPG/checkpoints/model_checkpoints/checkpoints/final_model.pth`).*

### 3. Launch GANDIVA
Double-click `run_gandiva.bat` or run:
```bash
python -m pharmpipe.web_bridge 3000
```
Then open your browser at:
`http://localhost:3000`

---

## Live Demo Workflow
1. Load `7ONS.pdb` (Human PARP1 complexed with inhibitor VKT, located in the project root) by clicking **"Upload File"** or typing `7ONS` in the PDB ID box and clicking **Fetch**.
2. Set candidate count (e.g. 4 or 6).
3. Click **"Generate candidates"**.
4. Inspect the 3D molecular structures, AutoDock Vina binding affinities ($-6.5$ to $-8.5\text{ kcal/mol}$), and QED drug-likeness.
5. Export results as `.sdf` or `.csv`.

---

## Architecture Overview
- **`pharmpipe/`**: Core 10-step pipeline (`steps/step0.py` to `step9.py`), adapters, gating logic, and live HTTP web bridge (`web_bridge.py`).
- **`frontend/`**: Interactive 3D WebGL user interface (`GANDIVA.html`).
- **`bin/vina.exe`**: Integrated AutoDock Vina v1.2.7 biophysical docking engine.
- **`cmdgen_repo/`**: Frozen GCPG transformer generator with stochastic multinomial sampling (`--sample`).
- **`pharmflow_repo/`**: Continuous vector-field Flow-Matching neural network (`PharmFlow`).
- **`fairseq_repo/`**: Fairseq transformer layers and configs.
