"""Summarize matched pharmacophore-sampler outputs against bound ligand features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from scipy.optimize import linear_sum_assignment


def reference_features(sdf_path: Path):
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"Could not parse reference ligand: {sdf_path}")
    factory = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    result = []
    for feat in factory.GetFeaturesForMol(mol):
        family = feat.GetFamily()
        if family in {"Aromatic", "Hydrophobe", "PosIonizable", "NegIonizable", "Acceptor", "Donor"}:
            result.append((family, np.asarray(feat.GetPos(), dtype=float)))
    return result


def one_sample_score(coords, class_ids, class_names, refs, threshold=2.0):
    pred = []
    for xyz, cls in zip(coords, class_ids):
        if int(cls) >= len(class_names):
            continue
        name = class_names[int(cls)]
        family = "Hydrophobe" if name == "LumpedHydrophobe" else name
        if family in {"Aromatic", "Hydrophobe", "PosIonizable", "NegIonizable", "Acceptor", "Donor"}:
            pred.append((family, np.asarray(xyz, dtype=float)))
    hit_pred = hit_ref = 0
    families = {family for family, _ in pred} | {family for family, _ in refs}
    for family in families:
        pred_xyz = [xyz for pfamily, xyz in pred if pfamily == family]
        ref_xyz = [xyz for rfamily, xyz in refs if rfamily == family]
        if not pred_xyz or not ref_xyz:
            continue
        distances = np.linalg.norm(np.asarray(pred_xyz)[:, None, :] - np.asarray(ref_xyz)[None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(distances)
        matches = int((distances[rows, cols] <= threshold).sum())
        hit_pred += matches
        hit_ref += matches
    precision = hit_pred / len(pred) if pred else 0.0
    recall = hit_ref / len(refs) if refs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n_scored_points": len(pred), "precision_at_2A": precision,
            "recall_at_2A": recall, "f1_at_2A": f1}


def diversity(coords, classes):
    values = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            cost = np.linalg.norm(coords[i, :, None, :] - coords[j, None, :, :], axis=-1)
            cost[classes[i, :, None] != classes[j, None, :]] += 20.0
            r, c = linear_sum_assignment(cost)
            values.append(float(np.sqrt(np.mean(cost[r, c] ** 2))))
    return float(np.mean(values)) if values else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--reference-sdf", type=Path, required=True)
    ap.add_argument("--diffphar", type=Path, required=True)
    ap.add_argument("--pharmflow", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    refs = reference_features(args.reference_sdf)
    rows, summary = [], {"target": args.target, "reference_feature_count": len(refs), "arms": {}}
    for arm, file in (("diffphar", args.diffphar), ("pharmflow", args.pharmflow)):
        data = np.load(file)
        coords, classes = data["coords"], data["class_indices"]
        names = data["class_names"].tolist()
        scores = [one_sample_score(x, c, names, refs) for x, c in zip(coords, classes)]
        for i, score in enumerate(scores):
            rows.append({"target": args.target, "arm": arm, "sample": i, **score})
        summary["arms"][arm] = {
            "n_samples": int(len(coords)),
            "valid_fraction": float(np.isfinite(coords).all(axis=(1, 2)).mean()),
            "mean_precision_at_2A": float(np.mean([s["precision_at_2A"] for s in scores])),
            "mean_recall_at_2A": float(np.mean([s["recall_at_2A"] for s in scores])),
            "mean_f1_at_2A": float(np.mean([s["f1_at_2A"] for s in scores])),
            "mean_type_aware_pairwise_cloud_distance_A": diversity(coords, classes),
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / f"{args.target}_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (args.out_dir / f"{args.target}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
