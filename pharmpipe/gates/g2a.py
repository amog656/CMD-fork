from ..data.models import ConsequenceType, GateOutcome, GateResult


def evaluate_g2a(*, trained: bool, config_hash: str) -> GateResult:
    reason = "PharmFlow not trained" if not trained else "matched DiffPhar evaluation data are missing"
    return GateResult("G2a", GateOutcome.NOT_EVALUABLE, ConsequenceType.DOWNGRADE, config_hash=config_hash, reason=reason)
