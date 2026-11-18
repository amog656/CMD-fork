from ..data.models import ConsequenceType, GateOutcome, GateResult


def evaluate_g3(*, auc_value: float | None, permutation_p: float | None, labelled_units: int, minimum_units: int, minimum_auc: float, maximum_p: float, config_hash: str) -> GateResult:
    if auc_value is None or permutation_p is None or labelled_units < minimum_units:
        return GateResult("G3", GateOutcome.NOT_EVALUABLE, ConsequenceType.DOWNGRADE, statistics={"auc": auc_value, "permutation_p": permutation_p}, sample_sizes={"labelled_units": labelled_units}, threshold=minimum_units, config_hash=config_hash, reason="labelled differential-feature evaluation is missing or below minimum")
    outcome = GateOutcome.PASS if auc_value >= minimum_auc and permutation_p < maximum_p else GateOutcome.FAIL
    return GateResult("G3", outcome, ConsequenceType.DOWNGRADE, statistics={"auc": auc_value, "permutation_p": permutation_p}, sample_sizes={"labelled_units": labelled_units}, threshold=minimum_auc, config_hash=config_hash, reason="differential-feature separability evaluation completed")
