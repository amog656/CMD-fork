from ..data.models import ConsequenceType, GateOutcome, GateResult


def evaluate_glfo(*, trained: bool, train_families: int, held_out_families: int, minimum_train_families: int, minimum_held_out_families: int, config_hash: str) -> GateResult:
    if not trained or train_families < minimum_train_families or held_out_families < minimum_held_out_families:
        reason = "PharmFlow not trained" if not trained else "family evidence is missing"
        if trained:
            reason = "family counts are below the preregistered train/held-out minima"
        return GateResult("G-LFO", GateOutcome.NOT_EVALUABLE, ConsequenceType.DOWNGRADE, sample_sizes={"train_families": train_families, "held_out_families": held_out_families}, config_hash=config_hash, reason=reason)
    return GateResult("G-LFO", GateOutcome.NOT_EVALUABLE, ConsequenceType.DOWNGRADE, sample_sizes={"train_families": train_families, "held_out_families": held_out_families}, config_hash=config_hash, reason="family-level PharmFlow evaluation statistics are unavailable")
