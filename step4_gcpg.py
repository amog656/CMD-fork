"""
step4_gcpg.py - Step 4: Gating Condition Pharmacophore Generation (GCPG).

Translates the consensus 3D pharmacophore blueprint into drug-like molecules (SMILES)
using the trained GCPG transformer model and RDKit 3D conformer generation.
"""

import sys
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class GCPGMoleculeGenerator:
    """Manages GCPG model execution and 3D conformer generation."""

    def __init__(
        self,
        gcpg_python: str = sys.executable,
        model_path: str = r"checkpoints_gcpg\model_checkpoints\checkpoints\final_model.pth",
        tokenizer_path: str = r"checkpoints_gcpg\model_checkpoints\checkpoints\tokenizer_r_iso.pkl",
        output_dir: str = "generated_molecules",
    ):
        self.gcpg_python = gcpg_python
        self.model_path = Path(model_path).resolve()
        self.tokenizer_path = Path(tokenizer_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_molecules(
        self,
        posp_path: str,
        n_mol: int = 50,
        mw_target: float = 400.0,
        logp_target: float = 3.5,
        qed_target: float = 0.65,
        seed: int = 42,
    ) -> Tuple[List[str], str]:
        """
        Executes GCPG generation script using the consensus .posp blueprint.
        Returns list of valid SMILES and path to the output text/SDF file.
        """
        if n_mol < 1:
            raise ValueError('n_mol must be at least 1')
        posp_file = Path(posp_path).resolve()
        if not posp_file.is_file():
            raise FileNotFoundError(f"Pharmacophore input does not exist: {posp_file}")
        if not self.model_path.is_file() or not self.tokenizer_path.is_file():
            raise FileNotFoundError(
                f"GCPG model/tokenizer missing: model={self.model_path}, tokenizer={self.tokenizer_path}"
            )
        if not Path(self.gcpg_python).is_file():
            raise FileNotFoundError(f"GCPG Python executable not found: {self.gcpg_python}")
        out_smiles_file = self.output_dir / f"{posp_file.stem}_generated_smiles.txt"
        out_sdf_file = self.output_dir / f"{posp_file.stem}_candidates_3d.sdf"

        # Embedded standalone generation runner to execute in the gcpg conda env
        runner_script = self.output_dir / "_run_gcpg.py"
        script_content = f"""
import sys
import os
import random
from pathlib import Path
import pickle
import torch
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

# Add gcpg_engine to path
sys.path.insert(0, str(Path(r"{Path(__file__).parent / 'gcpg_engine'}").resolve()))
import gcpg_utils.dataset as ds
sys.modules['utils'] = sys.modules['gcpg_utils']
sys.modules['utils.dataset'] = ds

from model.gcpg import GCPG
from gcpg_utils.file_utils import load_phar_file
import dgl

def load_model(model_path, tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    model_params = {{
        "max_len": 128,
        "cond_dim": 7,
        "pp_v_dim": 7 + 1,
        "pp_e_dim": 1,
        "pp_encoder_n_layer": 4,
        "hidden_dim": 384,
        "n_layers": 8,
        "ff_dim": 1024,
        "n_head": 8,
    }}
    model = GCPG(model_params, tokenizer)
    states = torch.load(model_path, map_location='cpu')
    sd = states.get('model', states)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch: {{len(missing)}} missing and {{len(unexpected)}} unexpected parameters")
    model.eval()
    return model, tokenizer

def main():
    random.seed({seed})
    np.random.seed({seed})
    torch.manual_seed({seed})
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all({seed})
    posp_path = Path(r"{posp_file}")
    model_path = Path(r"{self.model_path}")
    tokenizer_path = Path(r"{self.tokenizer_path}")
    out_file = Path(r"{out_smiles_file}")
    out_sdf = Path(r"{out_sdf_file}")

    print("[GCPG] Loading weights and tokenizer...")
    model, tokenizer = load_model(model_path, tokenizer_path)

    g = load_phar_file(posp_path)
    batch_size = 16
    g_batch = dgl.batch([g] * batch_size)

    # Condition tensors
    MW_t = torch.tensor([{mw_target}], dtype=torch.float32).repeat(1, batch_size)
    logP_t = torch.tensor([{logp_target}], dtype=torch.float32).repeat(1, batch_size)
    QED_t = torch.tensor([{qed_target}], dtype=torch.float32).repeat(1, batch_size)
    SAS_t = torch.tensor([3.0], dtype=torch.float32).repeat(1, batch_size)
    Rot_t = torch.tensor([5.0], dtype=torch.float32).repeat(1, batch_size)
    Score_t = torch.tensor([0.0], dtype=torch.float32).repeat(1, batch_size)
    Smi_t = torch.tensor([0.0], dtype=torch.float32).repeat(1, batch_size)
    conditions = [MW_t, logP_t, QED_t, SAS_t, Rot_t, Score_t, Smi_t]

    print(f"[GCPG] Generating {n_mol} molecules from consensus pharmacophore...")
    all_smiles = []
    n_batches = (({n_mol} + batch_size - 1) // batch_size)

    for b in range(n_batches):
        with torch.no_grad():
            gen_out = model.generate(pp_graphs=g_batch, conditions=conditions)
            texts = tokenizer.get_text(gen_out)
            all_smiles.extend(texts[:max(0, {n_mol} - len(all_smiles))])

    # Filter valid unique SMILES
    valid_mols = []
    seen = set()
    for smi in all_smiles:
        try:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                can_smi = Chem.MolToSmiles(m, isomericSmiles=True)
                if can_smi not in seen and len(can_smi) > 5:
                    seen.add(can_smi)
                    valid_mols.append((can_smi, m))
        except Exception:
            continue

    print(f"[GCPG] Successfully generated {{len(valid_mols)}} unique valid molecules.")

    # Generate 3D conformers and save SDF
    writer = Chem.SDWriter(str(out_sdf))
    embedded = []
    for i, (smi, m) in enumerate(valid_mols):
        try:
            m_h = Chem.AddHs(m)
            if AllChem.EmbedMolecule(m_h, randomSeed=42) < 0:
                continue
            AllChem.UFFOptimizeMolecule(m_h, maxIters=200)
            m_h.SetProp('_Name', f'candidate_{{i+1}}')
            m_h.SetProp('SMILES', smi)
            writer.write(m_h)
            embedded.append(smi)
        except Exception:
            continue
    writer.close()
    with open(out_file, 'w') as f:
        for smi in embedded:
            f.write(smi + '\\n')
    print(f"[GCPG] Wrote 3D conformers to {{out_sdf}}")

if __name__ == '__main__':
    main()
"""
        runner_script.write_text(script_content)

        print(f"[GCPG] Running generation in {self.gcpg_python}...")
        res = subprocess.run(
            [self.gcpg_python, str(runner_script)],
            capture_output=True,
            text=True
        )

        if res.returncode != 0:
            print("[GCPG Error]:", res.stderr)
            raise RuntimeError(f"GCPG generation failed with return code {res.returncode}")

        print(res.stdout)

        # Read back generated SMILES
        smiles_list = []
        if out_smiles_file.exists():
            smiles_list = [line.strip() for line in out_smiles_file.read_text().splitlines() if line.strip()]

        return smiles_list, str(out_sdf_file)
