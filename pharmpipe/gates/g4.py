from ..data.models import ConsequenceType, GateOutcome, GateResult


def evaluate_g4(*, auc_gain: float | None, minimum_auc_gain: float, adjusted_p_value: float | None, alpha: float, actives: int, decoys: int, minimum_actives: int, minimum_decoys: int, config_hash: str, reason: str = "") -> GateResult:
    if auc_gain is None or adjusted_p_value is None or actives < minimum_actives or decoys < minimum_decoys:
        return GateResult("G4", GateOutcome.NOT_EVALUABLE, ConsequenceType.STOP, statistics={"auc_gain": auc_gain, "adjusted_p_value": adjusted_p_value}, sample_sizes={"actives": actives, "decoys": decoys}, threshold=minimum_auc_gain, config_hash=config_hash, reason=reason or "retrospective sample is missing or below minimum")
    outcome = GateOutcome.PASS if auc_gain >= minimum_auc_gain and adjusted_p_value < alpha else GateOutcome.FAIL
    return GateResult("G4", outcome, ConsequenceType.STOP, statistics={"auc_gain": auc_gain, "adjusted_p_value": adjusted_p_value}, sample_sizes={"actives": actives, "decoys": decoys}, threshold=minimum_auc_gain, config_hash=config_hash, reason=reason or "primary Ensemble x PharmFlow versus Crystal x PharmFlow contrast evaluated")
