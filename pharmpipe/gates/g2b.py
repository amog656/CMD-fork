from ..data.models import ConsequenceType, GateOutcome, GateResult


def evaluate_g2b(*, auc_gain: float | None, ci_lower: float | None, minimum_ci_lower_bound: float, actives: int, decoys: int, minimum_actives: int, minimum_decoys: int, config_hash: str) -> GateResult:
    if auc_gain is None or ci_lower is None or actives < minimum_actives or decoys < minimum_decoys:
        return GateResult("G2b", GateOutcome.NOT_EVALUABLE, ConsequenceType.FALLBACK, statistics={"auc_gain": auc_gain, "ci_lower": ci_lower}, sample_sizes={"actives": actives, "decoys": decoys}, config_hash=config_hash, reason="retrospective sample is missing or below minimum")
    outcome = GateOutcome.PASS if ci_lower > minimum_ci_lower_bound else GateOutcome.FAIL
    return GateResult("G2b", outcome, ConsequenceType.FALLBACK, statistics={"auc_gain": auc_gain, "ci_lower": ci_lower}, sample_sizes={"actives": actives, "decoys": decoys}, threshold=minimum_ci_lower_bound, config_hash=config_hash, reason="ensemble versus crystal AUC difference evaluated")
