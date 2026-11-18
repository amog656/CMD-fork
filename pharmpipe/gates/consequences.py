"""Central gate consequence mapping and RunState transitions."""

from __future__ import annotations

from ..data.models import ConsequenceType, GateOutcome, GateResult, RunState

_CONSEQUENCES = {
    "G0": ConsequenceType.STOP,
    "Q": ConsequenceType.STOP,
    "G1": ConsequenceType.FALLBACK,
    "G2B": ConsequenceType.FALLBACK,
    "G2b": ConsequenceType.FALLBACK,
    "G-LFO": ConsequenceType.DOWNGRADE,
    "G2A": ConsequenceType.DOWNGRADE,
    "G2a": ConsequenceType.DOWNGRADE,
    "G3": ConsequenceType.DOWNGRADE,
    "D-state": ConsequenceType.DOWNGRADE,
    "G4": ConsequenceType.STOP,
    "FEP": ConsequenceType.DOWNGRADE,
    "G7-8": ConsequenceType.DOWNGRADE,
}
_CLAIMS = {
    "G-LFO": "family_generalization",
    "G2a": "pharmflow_over_diffphar",
    "G3": "differential_separability",
    "D-state": "state_specific_engagement",
    "FEP": "fep_supported_affinity_or_selectivity",
    "G7-8": "experimental_potency_or_selectivity",
}
_PASS_CLAIMS = {
    "G0": "target_applicability",
    "Q": "ensemble_physical_validity",
    "G1": "reference_state_coverage",
    "G2b": "ensemble_benefit",
    "G-LFO": "family_generalization",
    "G2a": "pharmflow_over_diffphar",
    "G3": "differential_separability",
    "D-state": "state_specific_engagement",
    "G4": "retrospective_ensemble_success",
    "FEP": "fep_supported_affinity_or_selectivity",
    "G7-8": "experimental_potency_or_selectivity",
}


def consequence_for_gate(gate_id: str) -> ConsequenceType:
    try:
        return _CONSEQUENCES[gate_id]
    except KeyError as exc:
        raise ValueError(f"unknown gate id: {gate_id}") from exc


def apply_consequence(gate_result: GateResult, run_state: RunState) -> RunState:
    expected = consequence_for_gate(gate_result.gate_id)
    if gate_result.consequence_type != expected:
        raise ValueError(f"{gate_result.gate_id} consequence must be {expected.value}")
    run_state.gate_log.append(gate_result)
    if gate_result.outcome == GateOutcome.PASS:
        claim = _PASS_CLAIMS.get(gate_result.gate_id)
        if claim:
            run_state.claims_allowed.add(claim)
        if gate_result.gate_id == "G4":
            run_state.confirmatory_generation_allowed = True
            run_state.exploratory_only = False
        return run_state

    gate_id = gate_result.gate_id
    if expected == ConsequenceType.STOP:
        if gate_id == "G4":
            run_state.confirmatory_generation_allowed = False
            run_state.use_crystal_baseline_only = True
            run_state.exploratory_only = True
            run_state.withdrawn_claims.add("retrospective_ensemble_success")
        elif gate_id == "Q":
            run_state.ensemble_branch_stopped = True
            run_state.ensemble_allowed = False
            run_state.use_crystal_baseline_only = True
            run_state.exploratory_only = True
        else:
            run_state.target_run_stopped = True
    elif gate_id == "G1":
        run_state.use_unweighted_states = True
        run_state.withdrawn_claims.add("population_weighted_state_claim")
    elif gate_id in {"G2b", "G2B"}:
        run_state.use_crystal_baseline_only = True
        run_state.withdrawn_claims.add("ensemble_benefit")
    elif gate_id in _CLAIMS:
        run_state.withdrawn_claims.add(_CLAIMS[gate_id])
    return run_state
