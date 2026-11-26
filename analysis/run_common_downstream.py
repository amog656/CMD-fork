"""Apply the same GMM, GCPG, conformer, and Vina stages to both sampler arms."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from step3_consensus import ConsensusPharmacophoreMerger
from step4_gcpg import GCPGMoleculeGenerator
from step5_docking import MultiStateDockingEngine


def molecule_properties(smiles):
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rows.append({
            "smiles": smi,
            "mw": float(Descriptors.MolWt(mol)),
            "qed": float(QED.qed(mol)),
            "logp": float(Crippen.MolLogP(mol)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        })
    return rows


def load_clouds(npz_path: Path):
    d = np.load(npz_path)
    coords, classes = d["coords"], d["class_indices"]
    labels = d["class_names"].tolist()
    return [{"coords": xyz, "types": [labels[int(i)] for i in cls]}
            for xyz, cls in zip(coords, classes)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["7NI4", "7ONS", "8A9K"])
    ap.add_argument("--samples-dir", type=Path, default=ROOT / "results" / "cmdgen_sampler_pilot")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "cmdgen_end_to_end_pilot")
    ap.add_argument("--n-molecules", type=int, default=50)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--exhaustiveness", type=int, default=1)
    ap.add_argument("--meeko-python", type=Path, default=Path(sys.executable))
    ap.add_argument("--gcpg-python", type=Path,
                    default=Path.home() / "miniconda3" / "envs" / "gcpg" / "python.exe")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for target in args.targets:
        receptor = ROOT / "baseline_assets" / "pilot_targets" / f"{target}_rec.pdb"
        ligand_pdb = ROOT / "baseline_assets" / "pilot_targets" / f"{target}_lig.pdb"
        ligand_model = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=True, sanitize=True)
        if ligand_model is None or ligand_model.GetNumAtoms() == 0:
            raise ValueError(f"Could not parse ligand center source: {ligand_pdb}")
        center = np.asarray(ligand_model.GetConformer().GetPositions(), dtype=float).mean(axis=0)

        engine = MultiStateDockingEngine(
            vina_exe=str(ROOT / "vina.exe"), meeko_python=str(args.meeko_python),
            allow_bad_receptor_residues=True,
            output_dir=str(args.out_dir / target / "shared_docking"),
        )
        receptor_pdbqt = engine.output_dir / f"{target}_receptor.pdbqt"
        engine.prepare_receptor_pdbqt(receptor, receptor_pdbqt)
        box_size = np.asarray([22.0, 22.0, 22.0])

        for arm in ("diffphar", "pharmflow"):
            arm_dir = args.out_dir / target / arm
            consensus_dir = arm_dir / "consensus"
            clouds = load_clouds(args.samples_dir / target / f"{arm}_seed11.npz")
            merger = ConsensusPharmacophoreMerger(str(consensus_dir))
            posp, blueprint, _ = merger.merge_states({"static": clouds}, n_consensus_features=7)
            feature_centers = np.asarray([item["center"] for item in blueprint["features"]], dtype=float)

            generator = GCPGMoleculeGenerator(gcpg_python=str(args.gcpg_python),
                                              output_dir=str(arm_dir / "gcpg"))
            smiles, sdf_path = generator.generate_molecules(
                posp_path=posp, n_mol=args.n_molecules, mw_target=400.0,
                logp_target=3.5, qed_target=0.65, seed=args.seed,
            )
            props = molecule_properties(smiles)
            properties_csv = arm_dir / "molecule_properties.csv"
            properties_csv.parent.mkdir(parents=True, exist_ok=True)
            with properties_csv.open("w", newline="", encoding="utf-8") as f:
                if props:
                    writer = csv.DictWriter(f, fieldnames=props[0].keys())
                    writer.writeheader(); writer.writerows(props)

            suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
            mols = [m for m in suppl if m is not None]
            dock_dir = arm_dir / "docking"
            dock_dir.mkdir(parents=True, exist_ok=True)
            dock_rows = []
            for idx, mol in enumerate(mols):
                name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"candidate_{idx+1}"
                ligand_pdbqt = dock_dir / f"{name}.pdbqt"
                if not engine.prepare_ligand_pdbqt(mol, ligand_pdbqt):
                    continue
                out_pose = dock_dir / f"{name}_out.pdbqt"
                affinity = engine.dock_molecule_single_state(
                    ligand_pdbqt, receptor_pdbqt, center, box_size, out_pose,
                    exhaustiveness=args.exhaustiveness,
                )
                dock_rows.append({"name": name,
                                  "smiles": mol.GetProp("SMILES") if mol.HasProp("SMILES") else Chem.MolToSmiles(mol),
                                  "vina_kcal_mol": float(affinity)})
            dock_rows.sort(key=lambda x: x["vina_kcal_mol"])
            with (arm_dir / "docking_scores.csv").open("w", newline="", encoding="utf-8") as f:
                if dock_rows:
                    writer = csv.DictWriter(f, fieldnames=dock_rows[0].keys())
                    writer.writeheader(); writer.writerows(dock_rows)

            score_values = [x["vina_kcal_mol"] for x in dock_rows]
            summary = {
                "target": target, "arm": arm, "seed": args.seed,
                "n_pharmacophore_samples": len(clouds),
                "consensus_features": len(blueprint["features"]),
                "requested_molecules": args.n_molecules,
                "valid_unique_smiles": len(smiles),
                "embedded_molecules": len(mols), "docked_molecules": len(dock_rows),
                "mean_mw": float(np.mean([p["mw"] for p in props])) if props else None,
                "mean_qed": float(np.mean([p["qed"] for p in props])) if props else None,
                "mean_logp": float(np.mean([p["logp"] for p in props])) if props else None,
                "mean_vina_kcal_mol": float(np.mean(score_values)) if score_values else None,
                "median_vina_kcal_mol": float(np.median(score_values)) if score_values else None,
                "best_vina_kcal_mol": float(min(score_values)) if score_values else None,
                "vina_exhaustiveness": args.exhaustiveness,
                "docking_box_center": center.tolist(), "docking_box_size": box_size.tolist(),
                "shared_gcpg_checkpoint": "checkpoints_gcpg/model_checkpoints/checkpoints/final_model.pth",
                "note": "Exploratory common-downstream pilot; docking scores are not experimental activity.",
            }
            (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            rows.append(summary)
            print(json.dumps(summary, indent=2), flush=True)

    with (args.out_dir / "all_arm_summaries.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
