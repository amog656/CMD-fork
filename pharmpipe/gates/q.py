from ..data.models import ConsequenceType, GateOutcome, GateResult


def evaluate_q(*, passed_count: int, total_count: int, minimum_fraction: float, config_hash: str, evidence_complete: bool = False) -> GateResult:
    if total_count <= 0 or passed_count < 0 or passed_count > total_count:
        return GateResult("Q", GateOutcome.NOT_EVALUABLE, ConsequenceType.STOP, sample_sizes={"conformers": total_count}, config_hash=config_hash, reason="ensemble QC sample is missing or invalid")
    fraction = passed_count / total_count
    if fraction < minimum_fraction:
        outcome = GateOutcome.FAIL
        reason = f"{passed_count} of {total_count} conformers passed configured geometry/clash QC"
    elif not evidence_complete:
        outcome = GateOutcome.NOT_EVALUABLE
        reason = "geometry/clash proxy passed, but restrained relaxation, physical energy, and aligned pocket RMSD evidence are unavailable"
    else:
        outcome = GateOutcome.PASS
        reason = f"{passed_count} of {total_count} conformers passed configured full QC"
    return GateResult("Q", outcome, ConsequenceType.STOP, statistics={"proxy_pass_fraction": fraction, "full_qc_evidence_available": evidence_complete}, sample_sizes={"conformers": total_count}, threshold=minimum_fraction, config_hash=config_hash, reason=reason)
