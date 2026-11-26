import sys
from pathlib import Path
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from step3_consensus import ConsensusPharmacophoreMerger
from step4_gcpg import GCPGMoleculeGenerator
from step5_docking import MultiStateDockingEngine

def load_clouds(npz_path: Path):
    d = np.load(npz_path)
    coords, classes = d["coords"], d["class_indices"]
    labels = d["class_names"].tolist()
    return [{"coords": xyz, "types": [labels[int(i)] for i in cls]} for xyz, cls in zip(coords, classes)]

def main():
    target = "AROE_THET8"
    receptor = ROOT / "baseline_assets" / "heldout_test_subset" / "AROE_THET8_1_263_0" / "2cy0_A_rec_2d5c_skm_lig_tt_min_0_pocket10.pdb"
    ligand_sdf = ROOT / "baseline_assets" / "heldout_test_subset" / "AROE_THET8_1_263_0" / "2cy0_A_rec_2d5c_skm_lig_tt_min_0.sdf"
    
    suppl = Chem.SDMolSupplier(str(ligand_sdf), removeHs=True, sanitize=False)
    mol = next((m for m in suppl if m is not None), None)
    center = np.asarray(mol.GetConformer().GetPositions(), dtype=float).mean(axis=0)

    out_base = ROOT / "results" / "heldout_comparison" / "end_to_end" / target
    out_base.mkdir(parents=True, exist_ok=True)

    engine = MultiStateDockingEngine(vina_exe=str(ROOT / "vina.exe"), output_dir=str(out_base / "shared_docking"))
    rec_pdbqt = engine.output_dir / f"{target}_rec.pdbqt"
    engine.prepare_receptor_pdbqt(receptor, rec_pdbqt)
    box_size = np.asarray([22.0, 22.0, 22.0])

    summaries = []
    for arm in ("diffphar", "pharmflow"):
        arm_dir = out_base / arm
        consensus_dir = arm_dir / "consensus"
        npz_file = ROOT / "results" / "heldout_comparison" / target / f"{arm}.npz"
        clouds = load_clouds(npz_file)

        merger = ConsensusPharmacophoreMerger(str(consensus_dir))
        posp, blueprint, _ = merger.merge_states({"static": clouds}, n_consensus_features=7)

        generator = GCPGMoleculeGenerator(
            gcpg_python=r"C:\Users\hanut sai\miniconda3\envs\gcpg\python.exe",
            output_dir=str(arm_dir / "gcpg")
        )
        smiles, sdf_path = generator.generate_molecules(posp_path=posp, n_mol=20, mw_target=400.0, logp_target=3.5, qed_target=0.65)

        # Docking
        mols = [m for m in Chem.SDMolSupplier(sdf_path, removeHs=False) if m is not None]
        scores = []
        for i, m in enumerate(mols):
            lig_pdbqt = engine.output_dir / f"{arm}_{i}.pdbqt"
            if engine.prepare_ligand_pdbqt(m, lig_pdbqt):
                out_dock = engine.output_dir / f"{arm}_{i}_out.pdbqt"
                val = engine.dock_molecule_single_state(lig_pdbqt, rec_pdbqt, center, box_size, out_dock, exhaustiveness=1)
                if val is not None:
                    scores.append(val)

        mws = [Descriptors.MolWt(m) for m in mols]
        qeds = [QED.qed(m) for m in mols]
        logps = [Crippen.MolLogP(m) for m in mols]

        summary = {
            "target": target,
            "arm": arm,
            "valid_unique": len(mols),
            "mean_mw": float(np.mean(mws)) if mws else 0.0,
            "mean_qed": float(np.mean(qeds)) if qeds else 0.0,
            "mean_logp": float(np.mean(logps)) if logps else 0.0,
            "docked_count": len(scores),
            "mean_vina": float(np.mean(scores)) if scores else 0.0,
            "median_vina": float(np.median(scores)) if scores else 0.0,
            "best_vina": float(np.min(scores)) if scores else 0.0,
        }
        summaries.append(summary)
        print(f"[{arm.upper()} Results]: {summary}")

    out_file = out_base / "summary.json"
    out_file.write_text(json.dumps(summaries, indent=2))
    print(f"Summary written to {out_file}")

if __name__ == "__main__":
    main()
