from ..data.models import ConsequenceType, GateOutcome, GateResult


def evaluate_d_state(*, potency_auc_drop: float | None, orthogonal_evidence: bool, config_hash: str) -> GateResult:
    if potency_auc_drop is None or not orthogonal_evidence:
        return GateResult("D-state", GateOutcome.NOT_EVALUABLE, ConsequenceType.DOWNGRADE, statistics={"potency_auc_drop": potency_auc_drop}, config_hash=config_hash, reason="held-out ablation or orthogonal state evidence is missing")
    return GateResult("D-state", GateOutcome.FAIL, ConsequenceType.DOWNGRADE, statistics={"potency_auc_drop": potency_auc_drop}, config_hash=config_hash, reason="full preregistered D-state confidence analysis is unavailable")
