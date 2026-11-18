from ..data.models import ConsequenceType, GateOutcome, GateResult


def evaluate_g0(*, usable_structure: bool, definable_pocket: bool, required_ligand_evidence: bool, config_hash: str, missing: list[str] | None = None) -> GateResult:
    missing = list(missing or [])
    passed = usable_structure and definable_pocket and required_ligand_evidence
    if not passed:
        outcome = GateOutcome.NOT_EVALUABLE if not usable_structure or not definable_pocket or not required_ligand_evidence else GateOutcome.FAIL
        reason = "required applicability evidence missing: " + ", ".join(missing)
    else:
        outcome, reason = GateOutcome.PASS, "structure, pocket, and required ligand evidence are available"
    return GateResult("G0", outcome, ConsequenceType.STOP, sample_sizes={}, config_hash=config_hash, reason=reason)
