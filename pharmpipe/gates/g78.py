from ..data.models import ConsequenceType, GateOutcome, GateResult


def evaluate_g78(*, synthesized: int, potency_hits: int, selective_hits: int, off_target_count: int, minimum_descriptive_n: int, config_hash: str) -> GateResult:
    if synthesized <= 0:
        return GateResult("G7-8", GateOutcome.NOT_EVALUABLE, ConsequenceType.DOWNGRADE, sample_sizes={"synthesized": synthesized}, config_hash=config_hash, reason="no synthesized assay-valid compounds")
    if synthesized < minimum_descriptive_n:
        return GateResult("G7-8", GateOutcome.NOT_EVALUABLE, ConsequenceType.DOWNGRADE, statistics={"potency_hits": potency_hits, "selective_hits": selective_hits}, sample_sizes={"synthesized": synthesized}, config_hash=config_hash, reason="below minimum sample size; hit rate is descriptive only")
    success = potency_hits > 0 and (off_target_count == 0 or selective_hits > 0)
    outcome = GateOutcome.PASS if success else GateOutcome.FAIL
    return GateResult("G7-8", outcome, ConsequenceType.DOWNGRADE, statistics={"potency_hits": potency_hits, "selective_hits": selective_hits}, sample_sizes={"synthesized": synthesized}, config_hash=config_hash, reason="experimental potency and selectivity criteria evaluated")
