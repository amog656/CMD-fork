from ..constants import RT_KCAL_PER_MOL_AT_REFERENCE_TEMPERATURE
from ..data.models import ConsequenceType, GateOutcome, GateResult
import math


def evaluate_fep(*, delta_delta_g: float | None, confidence_interval: tuple[float, float] | None, selectivity_goal_fold: float, replica_count: int, minimum_replicas: int, converged: bool, config_hash: str) -> GateResult:
    threshold = -RT_KCAL_PER_MOL_AT_REFERENCE_TEMPERATURE * math.log(selectivity_goal_fold)
    sizes = {"replicas": replica_count}
    if delta_delta_g is None or confidence_interval is None or replica_count < minimum_replicas or not converged:
        return GateResult("FEP", GateOutcome.NOT_EVALUABLE, ConsequenceType.DOWNGRADE, sample_sizes=sizes, threshold=threshold, config_hash=config_hash, reason="FEP data are missing, under-replicated, or unconverged")
    low, high = confidence_interval
    if low <= threshold < high:
        return GateResult("FEP", GateOutcome.NOT_EVALUABLE, ConsequenceType.DOWNGRADE, statistics={"delta_delta_G": delta_delta_g, "ci_low": low, "ci_high": high}, sample_sizes=sizes, threshold=threshold, config_hash=config_hash, reason="confidence interval crosses the preregistered threshold")
    outcome = GateOutcome.PASS if high <= threshold else GateOutcome.FAIL
    return GateResult("FEP", outcome, ConsequenceType.DOWNGRADE, statistics={"delta_delta_G": delta_delta_g, "ci_low": low, "ci_high": high}, sample_sizes=sizes, threshold=threshold, config_hash=config_hash, reason="FEP confidence interval evaluated against configured fold goal")
