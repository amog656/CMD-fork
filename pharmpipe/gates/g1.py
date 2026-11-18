from ..data.models import ConsequenceType, GateOutcome, GateResult


def evaluate_g1(*, reference_state_count: int, covered_state_count: int, minimum_coverage: float, config_hash: str) -> GateResult:
    if reference_state_count <= 0:
        return GateResult("G1", GateOutcome.NOT_EVALUABLE, ConsequenceType.FALLBACK, sample_sizes={"reference_states": reference_state_count}, config_hash=config_hash, reason="no reference states are available")
    coverage = covered_state_count / reference_state_count
    outcome = GateOutcome.PASS if coverage >= minimum_coverage else GateOutcome.FAIL
    return GateResult("G1", outcome, ConsequenceType.FALLBACK, statistics={"coverage": coverage}, sample_sizes={"reference_states": reference_state_count, "covered_states": covered_state_count}, threshold=minimum_coverage, config_hash=config_hash, reason=f"reference-state coverage is {coverage:.6g}")
