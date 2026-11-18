"""Step 9: MD mock pose stability and schema-checked wet-lab ingestion."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..adapters.md import MockMDAdapter
from ..config.loader import LoadedConfig
from ..data.models import GateOutcome, GateResult
from ..gates.g78 import evaluate_g78
from ..runio import prepare_stage_dir, write_manifest
from ..stats import wilson_interval


ASSAY_COLUMNS = {"compound_id", "protein", "endpoint", "value", "relation", "units", "assay_type", "assay_conditions", "replicate_id", "validity"}


def _criterion(value: float, relation: str, threshold: float) -> str:
    if relation == "=":
        return "hit" if value <= threshold else "miss"
    if relation == ">":
        return "miss" if value >= threshold else "indeterminate"
    if relation == "<":
        return "hit" if value <= threshold else "indeterminate"
    raise ValueError(f"unsupported assay relation: {relation}")


def _fold_assessment(target_row: dict, off_row: dict, goal_fold: float) -> dict:
    target_value, off_value = float(target_row["value"]), float(off_row["value"])
    if target_row["relation"] == "=" and off_row["relation"] == "=":
        ratio = off_value / target_value
        return {"status": "hit" if ratio >= goal_fold else "miss", "ratio": ratio, "relation": "="}
    if target_row["relation"] == "=" and off_row["relation"] == ">":
        bound = off_value / target_value
        return {"status": "hit" if bound >= goal_fold else "indeterminate", "ratio_lower_bound": bound, "relation": ">"}
    if target_row["relation"] == "=" and off_row["relation"] == "<":
        bound = off_value / target_value
        return {"status": "miss" if bound < goal_fold else "indeterminate", "ratio_upper_bound": bound, "relation": "<"}
    if target_row["relation"] == ">" and off_row["relation"] == "=":
        bound = off_value / target_value
        return {"status": "miss" if bound < goal_fold else "indeterminate", "ratio_upper_bound": bound, "relation": "<="}
    if target_row["relation"] == "<" and off_row["relation"] == "=":
        bound = off_value / target_value
        return {"status": "hit" if bound >= goal_fold else "indeterminate", "ratio_lower_bound": bound, "relation": ">="}
    return {"status": "indeterminate", "relation": "censored_pair"}


def run_step9(
    loaded: LoadedConfig, *, assay_csv: str | Path | None, compounds: list[dict], output_dir: str | Path,
    potency_goal: float, endpoint: str, expected_units: str, md_replicas: int,
    md_adapter: MockMDAdapter | None = None, minimum_descriptive_n: int, wilson_confidence: float,
) -> dict:
    if potency_goal <= 0 or md_replicas <= 0 or minimum_descriptive_n <= 0 or not 0 < wilson_confidence < 1:
        raise ValueError("potency goal and sample counts must be positive configured values")
    out = Path(output_dir)
    prepare_stage_dir(out)
    rows = []
    if assay_csv is not None:
        assay_path = Path(assay_csv)
        if not assay_path.is_file():
            raise FileNotFoundError(f"assay CSV does not exist: {assay_path}")
        with assay_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not ASSAY_COLUMNS.issubset(reader.fieldnames):
                raise ValueError(f"assay CSV is missing required columns: {sorted(ASSAY_COLUMNS - set(reader.fieldnames or []))}")
            rows = list(reader)
    valid_rows = [row for row in rows if row["validity"].strip().lower() in {"valid", "assay-valid", "true", "1"}]
    for row in valid_rows:
        if row["endpoint"] != endpoint or row["units"] != expected_units:
            raise ValueError("mixing assay endpoints or units is not allowed")
    by_compound: dict[str, dict[str, list[dict]]] = {}
    for row in valid_rows:
        by_compound.setdefault(row["compound_id"], {}).setdefault(row["protein"], []).append(row)
    potency_hits, selective_hits, labels = 0, 0, []
    assay_results = []
    target_name = loaded.config.target.name
    for compound in compounds:
        compound_id = compound["compound_id"]
        measurements = by_compound.get(compound_id, {})
        target_rows = measurements.get(target_name, [])
        target_results = [_criterion(float(row["value"]), row["relation"], potency_goal) for row in target_rows]
        potency_hit = bool(target_results) and all(result == "hit" for result in target_results)
        if not target_rows or "indeterminate" in target_results or ("hit" in target_results and "miss" in target_results):
            potency_label = "indeterminate"
        else:
            potency_label = "hit" if potency_hit else "miss"
        selective = potency_hit
        selectivity = {}
        for off_target in loaded.config.off_targets:
            off_rows = measurements.get(off_target.name, [])
            matched = [(target, off) for target in target_rows for off in off_rows if target["assay_type"] == off["assay_type"] and target["assay_conditions"] == off["assay_conditions"]]
            if not matched:
                selective = False
                selectivity[off_target.name] = {"status": "missing_or_unmatched_assay"}
                continue
            fold_results = [_fold_assessment(target, off, loaded.config.selectivity_goal_fold) for target, off in matched]
            statuses = [item["status"] for item in fold_results]
            if "miss" in statuses or "indeterminate" in statuses:
                selective = False
            selectivity[off_target.name] = {"status": "hit" if all(status == "hit" for status in statuses) else "miss" if "miss" in statuses else "indeterminate", "replicate_ratios": fold_results}
        potency_hits += int(potency_hit)
        selective_hits += int(selective)
        labels.append({"compound_id": compound_id, "potency": potency_label, "selective": selective, "selectivity": selectivity})
        assay_results.append({"compound_id": compound_id, "target_assay_rows": target_rows, "off_target_status": selectivity})
    synthesized = len(compounds)
    interval = wilson_interval(potency_hits, synthesized, confidence=wilson_confidence) if synthesized else None
    gate = evaluate_g78(synthesized=synthesized, potency_hits=potency_hits, selective_hits=selective_hits, off_target_count=len(loaded.config.off_targets), minimum_descriptive_n=minimum_descriptive_n, config_hash=loaded.config_hash)
    if not valid_rows:
        gate = GateResult("G7-8", GateOutcome.NOT_EVALUABLE, gate.consequence_type, config_hash=loaded.config_hash, reason="no assay-valid records are available")
    md = md_adapter or MockMDAdapter()
    md_results = [md.analyze_pose(item["compound_id"], protein, md_replicas, loaded.config.seed + idx) for idx, item in enumerate(compounds) for protein in [target_name, *[off.name for off in loaded.config.off_targets]]]
    (out / "assay_labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "assay_summary.json").write_text(json.dumps({"synthesized": synthesized, "potency_hits": potency_hits, "selective_hits": selective_hits, "potency_hit_rate_wilson_95": interval, "descriptive": synthesized < minimum_descriptive_n}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "md_pose_stability.json").write_text(json.dumps(md_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(out, config_hash=loaded.config_hash, seed=loaded.config.seed, counts_in={"assay_rows": len(rows), "compounds": synthesized}, counts_out={"valid_assay_rows": len(valid_rows), "compounds": synthesized}, dropped=[{"reason": "invalid_assay_record", "count": len(rows) - len(valid_rows)}])
    return {"gate": gate, "labels": labels, "wilson_interval": interval, "md_results": md_results}
