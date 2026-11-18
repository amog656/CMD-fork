"""Step 7: identical configurable alert, novelty, property, pose, and dedup filters."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..config.loader import LoadedConfig
from ..runio import prepare_stage_dir, write_manifest


def _catalogs():
    pains = FilterCatalog.FilterCatalogParams()
    pains.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    brenk = FilterCatalog.FilterCatalogParams()
    brenk.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(pains), FilterCatalog.FilterCatalog(brenk)


def run_step7(
    loaded: LoadedConfig, *, molecules_csv: str | Path, output_dir: str | Path,
    drop_alerts: bool, max_ecfp4_similarity: float | None, drop_known_scaffolds: bool,
    drop_property_failures: bool, exclusion_regions: list[dict] | None = None,
) -> dict:
    if max_ecfp4_similarity is not None and not 0 <= max_ecfp4_similarity <= 1:
        raise ValueError("max_ecfp4_similarity must be in [0,1]")
    out = Path(output_dir)
    prepare_stage_dir(out)
    with Path(molecules_csv).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pains, brenk = _catalogs()
    reference_mols, reference_fps, reference_scaffolds = [], [], set()
    active_path = loaded.config.reference.actives_path
    if active_path and Path(active_path).is_file():
        with Path(active_path).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                mol = Chem.MolFromSmiles(row.get("smiles", ""))
                if mol:
                    reference_mols.append(mol)
                    reference_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
                    reference_scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
    attrition = []
    kept, seen = [], set()
    for row in rows:
        reason = None
        mol = Chem.MolFromSmiles(row.get("smiles", ""))
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            reason = "invalid_or_fragmented"
        elif drop_alerts and (pains.HasMatch(mol) or brenk.HasMatch(mol)):
            reason = "structural_alert"
        elif drop_known_scaffolds and MurckoScaffold.MurckoScaffoldSmiles(mol=mol) in reference_scaffolds:
            reason = "known_scaffold"
        elif max_ecfp4_similarity is not None and reference_fps:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            if max(DataStructs.BulkTanimotoSimilarity(fp, reference_fps)) > max_ecfp4_similarity:
                reason = "known_analog"
        properties = json.loads(row.get("properties", "{}"))
        if reason is None and drop_property_failures:
            gates = loaded.config.property_gates.model_dump()
            if gates["MW"] is not None and properties.get("MW", 0) > gates["MW"]:
                reason = "property_gate_MW"
            elif gates["LogP"] is not None and properties.get("LogP", 0) > gates["LogP"]:
                reason = "property_gate_LogP"
            elif gates["QED"] is not None and properties.get("QED", 1) < gates["QED"]:
                reason = "property_gate_QED"
            elif gates["SAS"] is not None and properties.get("SAS", 0) > gates["SAS"]:
                reason = "property_gate_SAS"
        if reason is None and exclusion_regions:
            pose_paths = json.loads(row.get("pose_paths", "[]"))
            allowed_poses = [pose for pose in pose_paths if not _pose_file_in_regions(Path(molecules_csv).parent / pose, exclusion_regions)]
            if pose_paths and not allowed_poses:
                reason = "exclusion_volume_clash"
            elif pose_paths:
                row["pose_paths"] = json.dumps(allowed_poses)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True) if mol else ""
        if reason is None and canonical in seen:
            reason = "duplicate"
        if reason:
            attrition.append({"compound_id": row.get("compound_id"), "reason": reason})
        else:
            seen.add(canonical)
            row["canonical_smiles"] = canonical
            kept.append(row)
    filtered_path = out / "filtered_molecules.csv"
    fieldnames = list(rows[0]) + (["canonical_smiles"] if rows and "canonical_smiles" not in rows[0] else []) if rows else ["compound_id", "smiles", "canonical_smiles"]
    with filtered_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    attrition_path = out / "attrition.json"
    attrition_path.write_text(json.dumps({"input_count": len(rows), "output_count": len(kept), "dropped_count": len(attrition), "dropped": attrition}, indent=2) + "\n", encoding="utf-8")
    write_manifest(out, config_hash=loaded.config_hash, seed=loaded.config.seed, counts_in={"molecules": len(rows)}, counts_out={"molecules": len(kept)}, dropped=attrition)
    return {"filtered_molecules": filtered_path, "attrition": attrition_path, "kept": kept}


def _pose_in_regions(pose_xyz: str, regions: list[dict]) -> bool:
    import numpy as np
    coordinates = np.asarray(json.loads(pose_xyz), dtype=float)
    for region in regions:
        center = np.asarray(region["center_A"], dtype=float)
        if np.any(np.linalg.norm(coordinates - center, axis=1) <= float(region["radius_A"])):
            return True
    return False


def _pose_file_in_regions(path: str, regions: list[dict]) -> bool:
    import numpy as np
    molecule = next(Chem.SDMolSupplier(path, removeHs=True), None)
    if molecule is None or molecule.GetNumConformers() == 0:
        return True
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    for region in regions:
        center = np.asarray(region["center_A"], dtype=float)
        if np.any(np.linalg.norm(coordinates - center, axis=1) <= float(region["radius_A"])):
            return True
    return False
