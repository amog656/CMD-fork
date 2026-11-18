"""Step 0: validate applicability and write the resolved target inputs."""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path

from ..config.loader import LoadedConfig
from ..data.models import GateResult
from ..gates.g0 import evaluate_g0
from ..runio import prepare_stage_dir, write_manifest


@dataclass
class Step0Result:
    gate: GateResult
    pocket_residues: list[str]
    applicability_report: dict


def _parse_pdb(path: Path) -> list[dict]:
    atoms = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 54:
                try:
                    atoms.append({
                        "atom": line[12:16].strip(), "residue": line[17:20].strip(),
                        "chain": line[21:22].strip(), "resid": line[22:26].strip(),
                        "element": line[76:78].strip() or line[12:16].strip()[0],
                        "xyz": [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    })
                except ValueError:
                    continue
    return atoms


def _count_rows(path: str | None) -> int:
    if not path or not Path(path).is_file():
        return 0
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def run_step0(
    loaded: LoadedConfig, *, target_structure: str | Path | None, output_dir: str | Path,
) -> Step0Result:
    out = Path(output_dir)
    prepare_stage_dir(out)
    structure_path = Path(target_structure or loaded.config.target.structure_path or "")
    missing = []
    atoms = _parse_pdb(structure_path) if structure_path.is_file() else []
    if not atoms:
        missing.append("usable target structure")
    construct = loaded.config.target.construct_info
    chain_atoms = []
    for atom in atoms:
        try:
            residue_number = int(atom["resid"])
        except ValueError:
            continue
        if atom["chain"] == construct.chain and construct.residues[0] <= residue_number <= construct.residues[1]:
            chain_atoms.append(atom)
    if not chain_atoms:
        missing.append("construct residues in target structure")
    available_residues = {a["resid"] for a in chain_atoms}
    requested = loaded.config.target.pocket_residues
    if loaded.config.target.pocket_source == "residues":
        if not requested:
            missing.append("pocket residue definition")
            pocket_residues = []
        else:
            pocket_residues = [str(item) for item in requested if str(item) in available_residues]
            if not pocket_residues:
                missing.append("resolved pocket residues")
    else:
        pocket_residues = []
    active_path = loaded.config.reference.actives_path
    active_count = _count_rows(active_path)
    inactives_count = _count_rows(loaded.config.reference.inactives_path)
    decoy_count = _count_rows(loaded.config.reference.decoys_path)
    ligand_evidence = active_count > 0
    if not ligand_evidence:
        missing.append("reference actives file")
    gate = evaluate_g0(usable_structure=bool(atoms and chain_atoms), definable_pocket=bool(pocket_residues or loaded.config.target.pocket_source == "detector"), required_ligand_evidence=ligand_evidence, config_hash=loaded.config_hash, missing=missing)
    report = {
        "target": loaded.config.target.name,
        "structure_path": str(structure_path) if atoms else None,
        "atom_count": len(atoms), "construct_atom_count": len(chain_atoms),
        "pocket_source": loaded.config.target.pocket_source, "pocket_residues": pocket_residues,
        "reference_actives_present": ligand_evidence,
        "reference_counts": {"actives": active_count, "inactives": inactives_count, "decoys": decoy_count},
        "reference_decoys_present": decoy_count > 0,
        "missing_required": missing,
        "premarked_gates": ({"G4": "decoys missing", "G2b": "decoys missing", **({"G3": "decoys/labelled units missing"} if loaded.config.off_targets else {})} if decoy_count == 0 else ({"G3": "selectivity analysis skipped for Mode A"} if loaded.config.mode == "A" else {})),
        "gate": gate.outcome.value,
    }
    (out / "applicability_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "pocket_definition.json").write_text(json.dumps({"residues": pocket_residues, "source": loaded.config.target.pocket_source}, indent=2) + "\n", encoding="utf-8")
    write_manifest(out, config_hash=loaded.config_hash, seed=loaded.config.seed, counts_in={"structure_atoms": len(atoms)}, counts_out={"pocket_residues": len(pocket_residues)}, dropped=[])
    return Step0Result(gate, pocket_residues, report)
