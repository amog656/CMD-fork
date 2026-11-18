"""Gate-log-derived report; stage metrics are never accepted as claims directly."""

from __future__ import annotations

import json
from pathlib import Path

from ..data.models import RunState


def generate_summary(gate_log_path: str | Path, run_state: RunState, output_path: str | Path) -> Path:
    gate_path = Path(gate_log_path)
    records = []
    if gate_path.exists():
        records = [json.loads(line) for line in gate_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    passing_ids = {record["gate_id"] for record in records if record["outcome"] == "PASS"}
    claim_gate = {
        "target_applicability": "G0", "ensemble_physical_validity": "Q", "reference_state_coverage": "G1",
        "ensemble_benefit": "G2b", "family_generalization": "G-LFO", "pharmflow_over_diffphar": "G2a",
        "differential_separability": "G3", "state_specific_engagement": "D-state",
        "retrospective_ensemble_success": "G4", "fep_supported_affinity_or_selectivity": "FEP",
        "experimental_potency_or_selectivity": "G7-8",
    }
    allowed = sorted(claim for claim in run_state.claims_allowed if claim_gate.get(claim) in passing_ids)
    withdrawn = sorted(run_state.withdrawn_claims)
    lines = ["# Pipeline summary", "", "## Gate outcomes", ""]
    if records:
        for record in records:
            lines.append(f"- **{record['gate_id']} — {record['outcome']}** ({record['consequence_type']}): {record['reason']}")
    else:
        lines.append("No gate results were recorded.")
    lines.extend(["", "## Claims allowed", ""])
    lines.extend([f"- {claim}" for claim in allowed] or ["- None"])
    lines.extend(["", "## Claims withdrawn", ""])
    lines.extend([f"- {claim}" for claim in withdrawn] or ["- None"])
    if run_state.premarked_gates:
        lines.extend(["", "## Not run or premarked", ""])
        lines.extend([f"- **{gate_id}:** {reason}" for gate_id, reason in sorted(run_state.premarked_gates.items())])
    if run_state.target_run_stopped:
        lines.extend(["", "Target run stopped by a STOP gate."])
    elif run_state.ensemble_branch_stopped:
        lines.extend(["", "Ensemble branch stopped; downstream generation is exploratory."])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
