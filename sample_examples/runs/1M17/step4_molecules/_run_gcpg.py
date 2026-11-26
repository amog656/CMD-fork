
import sys
import os
from pathlib import Path
import pickle
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

# Add gcpg_engine to path
sys.path.insert(0, str(Path(r"C:\Users\hanut sai\Desktop\nnn\gcpg_engine").resolve()))
import gcpg_utils.dataset as ds
sys.modules['utils'] = sys.modules['gcpg_utils']
sys.modules['utils.dataset'] = ds

from model.gcpg import GCPG
from gcpg_utils.file_utils import load_phar_file
import dgl

def load_model(model_path, tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    model_params = {
        "max_len": 128,
        "cond_dim": 7,
        "pp_v_dim": 7 + 1,
        "pp_e_dim": 1,
        "pp_encoder_n_layer": 4,
        "hidden_dim": 384,
        "n_layers": 8,
        "ff_dim": 1024,
        "n_head": 8,
    }
    model = GCPG(model_params, tokenizer)
    states = torch.load(model_path, map_location='cpu')
    sd = states.get('model', states)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch: {len(missing)} missing and {len(unexpected)} unexpected parameters")
    model.eval()
    return model, tokenizer

def main():
    posp_path = Path(r"C:\Users\hanut sai\Desktop\nnn\sample_examples\runs\1M17\step3_consensus\consensus_blueprint.posp")
    model_path = Path(r"C:\Users\hanut sai\Desktop\nnn\checkpoints_gcpg\model_checkpoints\checkpoints\final_model.pth")
    tokenizer_path = Path(r"C:\Users\hanut sai\Desktop\nnn\checkpoints_gcpg\model_checkpoints\checkpoints\tokenizer_r_iso.pkl")
    out_file = Path(r"C:\Users\hanut sai\Desktop\nnn\sample_examples\runs\1M17\step4_molecules\consensus_blueprint_generated_smiles.txt")
    out_sdf = Path(r"C:\Users\hanut sai\Desktop\nnn\sample_examples\runs\1M17\step4_molecules\consensus_blueprint_candidates_3d.sdf")

    print("[GCPG] Loading weights and tokenizer...")
    model, tokenizer = load_model(model_path, tokenizer_path)

    g = load_phar_file(posp_path)
    batch_size = 16
    g_batch = dgl.batch([g] * batch_size)

    # Condition tensors
    MW_t = torch.tensor([400.0], dtype=torch.float32).repeat(1, batch_size)
    logP_t = torch.tensor([3.5], dtype=torch.float32).repeat(1, batch_size)
    QED_t = torch.tensor([0.65], dtype=torch.float32).repeat(1, batch_size)
    SAS_t = torch.tensor([3.0], dtype=torch.float32).repeat(1, batch_size)
    Rot_t = torch.tensor([5.0], dtype=torch.float32).repeat(1, batch_size)
    Score_t = torch.tensor([0.0], dtype=torch.float32).repeat(1, batch_size)
    Smi_t = torch.tensor([0.0], dtype=torch.float32).repeat(1, batch_size)
    conditions = [MW_t, logP_t, QED_t, SAS_t, Rot_t, Score_t, Smi_t]

    print(f"[GCPG] Generating 2 molecules from consensus pharmacophore...")
    all_smiles = []
    n_batches = ((2 + batch_size - 1) // batch_size)

    for b in range(n_batches):
        with torch.no_grad():
            gen_out = model.generate(pp_graphs=g_batch, conditions=conditions)
            texts = tokenizer.get_text(gen_out)
            all_smiles.extend(texts[:max(0, 2 - len(all_smiles))])

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

    print(f"[GCPG] Successfully generated {len(valid_mols)} unique valid molecules.")

    # Generate 3D conformers and save SDF
    writer = Chem.SDWriter(str(out_sdf))
    embedded = []
    for i, (smi, m) in enumerate(valid_mols):
        try:
            m_h = Chem.AddHs(m)
            if AllChem.EmbedMolecule(m_h, randomSeed=42) < 0:
                continue
            AllChem.UFFOptimizeMolecule(m_h, maxIters=200)
            m_h.SetProp('_Name', f'candidate_{i+1}')
            m_h.SetProp('SMILES', smi)
            writer.write(m_h)
            embedded.append(smi)
        except Exception:
            continue
    writer.close()
    with open(out_file, 'w') as f:
        for smi in embedded:
            f.write(smi + '\n')
    print(f"[GCPG] Wrote 3D conformers to {out_sdf}")

if __name__ == '__main__':
    main()
