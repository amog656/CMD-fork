"""Step 6: seeded mock generation, RDKit validation, properties, and pose fitting."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit import RDConfig
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ..adapters.gcpg import MockGCPGAdapter
from ..config.loader import LoadedConfig
from ..data.models import Molecule, PharmModel, PharmPoint, RunState
from ..metrics import compute_properties
from ..runio import prepare_stage_dir, write_manifest
from .pharmacophore_graph import FEATURE_TYPE_MAPPING, pharmacophore_graph


@dataclass
class Step6Result:
    molecules: list[Molecule]
    molecules_csv: Path
    property_table: Path
    pose_paths: list[Path]


def _feature_factory():
    path = str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    return ChemicalFeatures.BuildFeatureFactory(path)


def _ligand_features(mol, conf_id: int) -> list[tuple[str, np.ndarray]]:
    conformer = mol.GetConformer(conf_id)
    out = []
    for feature in _feature_factory().GetFeaturesForMol(mol):
        feature_type = FEATURE_TYPE_MAPPING.get(feature.GetFamily(), "OTHERS")
        atoms = feature.GetAtomIds()
        xyz = np.mean([np.asarray(conformer.GetAtomPosition(atom)) for atom in atoms], axis=0)
        out.append((feature_type, xyz))
    return out


def _best_alignment(mol, conf_id: int, points: list[PharmPoint], match_distance_A: float):
    features = _ligand_features(mol, conf_id)
    if not features or not points:
        return None
    ligand_xyz = np.asarray([item[1] for item in features], dtype=float)
    target_xyz = np.asarray([point.xyz for point in points], dtype=float)
    distances = cdist(ligand_xyz, target_xyz)
    same_type = np.asarray([[feature[0] == point.type for point in points] for feature in features])
    rows, cols = linear_sum_assignment(np.where(same_type, distances, np.finfo(float).max / 100))
    accepted = [(r, c) for r, c in zip(rows, cols) if same_type[r, c] and distances[r, c] <= match_distance_A]
    if not accepted:
        return None
    source = ligand_xyz[[r for r, _ in accepted]]
    target = target_xyz[[c for _, c in accepted]]
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    conf = mol.GetConformer(conf_id)
    for atom_index in range(mol.GetNumAtoms()):
        position = np.asarray(conf.GetAtomPosition(atom_index))
        transformed = (position - source_center) @ rotation + target_center
        conf.SetAtomPosition(atom_index, transformed.tolist())
    match_score = len(accepted) / max(len(features), len(points))
    unmatched = len(points) - len(accepted)
    return float(match_score), int(unmatched)


def _property_pass(properties: dict, requests: dict) -> bool:
    directions = {"MW": "max", "LogP": "max", "QED": "min", "SAS": "max", "rotatable_bonds": "max", "TPSA": "max"}
    for key, requested in requests.items():
        if requested is None:
            continue
        value = properties[key]
        if directions[key] == "max" and value > requested:
            return False
        if directions[key] == "min" and value < requested:
            return False
    return True


def run_step6(
    loaded: LoadedConfig, *, run_state: RunState, models: list[PharmModel], crystal_baseline_model: PharmModel | None,
    output_dir: str | Path, count_per_model: int, repeats: int, conformer_count: int,
    edge_distance_mode: str, match_distance_A: float, max_unmatched_points: int,
    adapter: MockGCPGAdapter | None = None,
) -> Step6Result:
    if min(count_per_model, repeats, conformer_count) <= 0 or match_distance_A <= 0 or max_unmatched_points < 0:
        raise ValueError("generation counts and matching limits must be valid configured values")
    exploratory = not run_state.confirmatory_generation_allowed
    if exploratory:
        if crystal_baseline_model is None:
            raise ValueError("exploratory generation requires a crystal-baseline pharmacophore")
        selected_models = [crystal_baseline_model]
    else:
        selected_models = list(models)
    if not selected_models:
        raise ValueError("no pharmacophore models were selected for generation")
    for model in selected_models:
        pharmacophore_graph(model, edge_distance_mode=edge_distance_mode)
    out = Path(output_dir)
    prepare_stage_dir(out)
    poses_dir = out / "poses"
    poses_dir.mkdir()
    generator = adapter or MockGCPGAdapter()
    molecules: dict[str, Molecule] = {}
    molecule_poses: dict[str, list[str]] = {}
    poses: list[Path] = []
    invalid_count = 0
    aligned_count = 0
    for model_index, model in enumerate(selected_models):
        for repeat in range(repeats):
            seed = loaded.config.seed + model_index * repeats + repeat
            generated = generator.generate(model, count_per_model, loaded.config.property_gates.model_dump(), seed)
            for generated_index, smiles in enumerate(generated):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    invalid_count += 1
                    continue
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                if canonical in molecules:
                    continue
                properties = compute_properties(canonical)
                molecule_seed = seed + generated_index
                molecule = Molecule(canonical, model.model_id, properties, {"valid": True, "exploratory": exploratory, "property_request_met": _property_pass(properties, loaded.config.property_gates.model_dump()), "passed_filters": False}, seed=molecule_seed)
                molecules[canonical] = molecule
                mol = Chem.AddHs(mol)
                params = AllChem.ETKDGv3()
                params.randomSeed = int(molecule_seed)
                conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=conformer_count, params=params))
                for conformer_id in conformer_ids:
                    match = _best_alignment(mol, conformer_id, model.points, match_distance_A)
                    if match is None or match[1] > max_unmatched_points:
                        continue
                    score, unmatched = match
                    molecule.aggregated_scores["pharmacophore_match"] = max(score, molecule.aggregated_scores.get("pharmacophore_match", 0.0))
                    molecule.unmatched_pharmacophore_points = min(unmatched, molecule.unmatched_pharmacophore_points if molecule.unmatched_pharmacophore_points is not None else unmatched)
                    filename = f"pose_{len(poses):06d}.sdf"
                    path = poses_dir / filename
                    writer = Chem.SDWriter(str(path))
                    writer.write(mol, confId=conformer_id)
                    writer.close()
                    poses.append(path)
                    molecule_poses.setdefault(canonical, []).append(str(Path("poses") / filename))
                    aligned_count += 1
    rows = []
    for index, molecule in enumerate(molecules.values()):
        rows.append({"compound_id": f"mol_{index:06d}", "smiles": molecule.smiles, "hypothesis_id": molecule.source_model_id, "seed": molecule.seed, "properties": json.dumps(molecule.properties, sort_keys=True), "match_score": molecule.aggregated_scores.get("pharmacophore_match"), "unmatched_points": molecule.unmatched_pharmacophore_points, "pose_paths": json.dumps(molecule_poses.get(molecule.smiles, [])), "exploratory": molecule.flags["exploratory"], "valid": molecule.flags["valid"]})
    table = out / "molecules.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["compound_id", "smiles", "hypothesis_id", "seed", "properties", "match_score", "unmatched_points", "pose_paths", "exploratory", "valid"])
        writer.writeheader()
        writer.writerows(rows)
    prop_table = out / "property_achievement.csv"
    with prop_table.open("w", newline="", encoding="utf-8") as handle:
        fields = ["smiles", *loaded.config.property_gates.model_dump().keys(), *[f"achieved_{key}" for key in loaded.config.property_gates.model_dump().keys()], "property_request_met"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for molecule in molecules.values():
            row = {"smiles": molecule.smiles, "property_request_met": molecule.flags["property_request_met"]}
            for key, request in loaded.config.property_gates.model_dump().items():
                row[key] = request
                row[f"achieved_{key}"] = molecule.properties[key]
            writer.writerow(row)
    manifest = {"config_hash": loaded.config_hash, "seed": loaded.config.seed, "generated": count_per_model * repeats * len(selected_models), "valid_unique": len(molecules), "invalid": invalid_count, "aligned_poses": aligned_count, "exploratory": exploratory, "selected_hypotheses": [model.model_id for model in selected_models]}
    (out / "generation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(out, config_hash=loaded.config_hash, seed=loaded.config.seed, counts_in={"requested_generation": manifest["generated"]}, counts_out={"valid_unique": len(molecules), "aligned_poses": aligned_count}, dropped=[{"reason": "invalid_smiles", "count": invalid_count}])
    return Step6Result(list(molecules.values()), table, prop_table, poses)
