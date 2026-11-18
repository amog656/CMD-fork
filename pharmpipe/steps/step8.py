"""Step 8: per-protein docking, MM-GBSA triage, and FEP shortlist."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..adapters.docking import MockDockingAdapter
from ..adapters.fep import MockFEPAdapter
from ..adapters.mmgbsa import MockMMGBSAAdapter
from ..config.loader import LoadedConfig
from ..data.models import ConsequenceType, GateOutcome, GateResult
from ..gates.fep import evaluate_fep
from ..runio import prepare_stage_dir, write_manifest


def run_step8(
    loaded: LoadedConfig, *, filtered_molecules_csv: str | Path, target_states: list[str],
    off_target_states: dict[str, list[str]], output_dir: str | Path,
    mmgbsa_top_count: int, minimum_fep_replicas: int, docking: MockDockingAdapter | None = None,
    mmgbsa: MockMMGBSAAdapter | None = None, fep: MockFEPAdapter | None = None,
) -> dict:
    if not target_states or mmgbsa_top_count <= 0 or minimum_fep_replicas <= 0:
        raise ValueError("target states and positive funnel counts are required")
    missing_off_targets = {item.name for item in loaded.config.off_targets} - set(off_target_states)
    out = Path(output_dir)
    prepare_stage_dir(out)
    with Path(filtered_molecules_csv).open("r", newline="", encoding="utf-8") as handle:
        molecules = list(csv.DictReader(handle))
    dock_adapter, mm_adapter, fep_adapter = docking or MockDockingAdapter(), mmgbsa or MockMMGBSAAdapter(), fep or MockFEPAdapter()
    per_protein = {loaded.config.target.name: target_states, **off_target_states}
    docking_records = []
    target_rank = []
    for protein_index, (protein, states) in enumerate(per_protein.items()):
        if not states:
            raise ValueError(f"protein {protein!r} has no retained pocket states")
        for molecule_index, row in enumerate(molecules):
            state_scores = [(state_id, dock_adapter.score(row["smiles"], state_id, loaded.config.seed + molecule_index + slot)) for slot, state_id in enumerate(states)]
            ordered = sorted(state_scores, key=lambda item: item[1])
            summary = {"protein": protein, "compound_id": row.get("compound_id"), "best_score": ordered[0][1], "best_state_id": ordered[0][0], "median_score": float(np.median([score for _, score in state_scores])), "score_range": float(max(score for _, score in state_scores) - min(score for _, score in state_scores)), "K": len(states)}
            docking_records.append(summary)
            if protein_index == 0:
                target_rank.append((summary["best_score"], row))
    target_rank.sort(key=lambda item: item[0])
    mm_records = []
    for rank, (_, row) in enumerate(target_rank[:mmgbsa_top_count]):
        score, uncertainty = mm_adapter.rescore(row["smiles"], row.get("compound_id", ""), loaded.config.seed + rank)
        mm_records.append({"compound_id": row.get("compound_id"), "smiles": row["smiles"], "mm_gbsa": score, "uncertainty": uncertainty})
    mm_records.sort(key=lambda record: record["mm_gbsa"])
    shortlist = mm_records[:loaded.config.fep_shortlist_size]
    fep_records = []
    for index, row in enumerate(shortlist):
        protein_ids = [loaded.config.target.name, *off_target_states.keys()]
        for protein_index, protein in enumerate(protein_ids):
            fep_records.append({"compound_id": row["compound_id"], "protein": protein, **fep_adapter.calculate(row["smiles"], protein, minimum_fep_replicas, loaded.config.seed + index + protein_index)})
    fep_gate = evaluate_fep(delta_delta_g=None, confidence_interval=None, selectivity_goal_fold=loaded.config.selectivity_goal_fold, replica_count=0, minimum_replicas=minimum_fep_replicas, converged=False, config_hash=loaded.config_hash)
    if missing_off_targets:
        fep_gate.reason = "FEP cannot evaluate selectivity because off-target state sets are missing: " + ", ".join(sorted(missing_off_targets))
    if not molecules:
        fep_gate = GateResult("FEP", GateOutcome.NOT_EVALUABLE, ConsequenceType.DOWNGRADE, config_hash=loaded.config_hash, reason="no filtered molecules reached the scoring funnel")
    def write_records(filename, records):
        path = out / filename
        path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
    docking_path = write_records("docking_by_protein.json", docking_records)
    mm_path = write_records("mmgbsa.json", mm_records)
    fep_path = write_records("fep.json", fep_records)
    attrition_path = write_records("funnel_attrition.json", {"filtered": len(molecules), "docked": len(target_rank), "mmgbsa": len(mm_records), "fep_shortlist": len(shortlist)})
    write_manifest(out, config_hash=loaded.config_hash, seed=loaded.config.seed, counts_in={"filtered_molecules": len(molecules)}, counts_out={"docked": len(target_rank), "mmgbsa": len(mm_records), "fep_shortlist": len(shortlist)}, dropped=[])
    return {"docking": docking_path, "mmgbsa": mm_path, "fep": fep_path, "attrition": attrition_path, "gate": fep_gate}
