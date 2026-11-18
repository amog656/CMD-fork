"""Shared typed records used across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal


class GateOutcome(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class ConsequenceType(StrEnum):
    STOP = "STOP"
    FALLBACK = "FALLBACK"
    DOWNGRADE = "DOWNGRADE"


PharmType = Literal["AROM", "HYBL", "POSC", "NEGI", "HACC", "HDON", "LHYBL", "OTHERS"]


@dataclass
class PocketAtom:
    xyz: tuple[float, float, float]
    element: str
    residue_name: str
    residue_id: str
    atom_name: str


@dataclass
class PocketState:
    state_id: str
    source: Literal["apo-seeded", "holo-seeded", "ligand-bound", "crystal"]
    coordinate_path: Path
    pocket_atoms: list[PocketAtom] = field(default_factory=list)
    feature_matrix: list[list[float]] = field(default_factory=list)
    qc_flags: dict[str, bool] = field(default_factory=dict)
    cluster_id: str | None = None
    weight: float | None = None
    in_domain: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    alignment_transform: list[list[float]] | None = None


@dataclass
class PharmPoint:
    xyz: tuple[float, float, float]
    type: PharmType
    confidence: float | None = None


@dataclass
class PharmModel:
    model_id: str
    points: list[PharmPoint]
    origin: Literal["per-state", "consensus", "differential", "state-specific"]
    supporting_state_ids: list[str] = field(default_factory=list)
    vote_counts: list[int] = field(default_factory=list)


@dataclass
class Molecule:
    smiles: str
    source_model_id: str
    properties: dict[str, float | int | None] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    docking_scores_by_state: dict[str, float] = field(default_factory=dict)
    aggregated_scores: dict[str, float] = field(default_factory=dict)
    mm_gbsa: float | None = None
    mm_gbsa_uncertainty: float | None = None
    fep: float | None = None
    fep_uncertainty: float | None = None
    seed: int | None = None
    unmatched_pharmacophore_points: int | None = None


@dataclass
class GateResult:
    gate_id: str
    outcome: GateOutcome
    consequence_type: ConsequenceType
    statistics: dict[str, float | int | str | None] = field(default_factory=dict)
    sample_sizes: dict[str, int] = field(default_factory=dict)
    threshold: float | int | str | None = None
    config_hash: str = ""
    reason: str = ""


@dataclass
class RunState:
    mode: str = "A"
    surviving_states: list[PocketState] = field(default_factory=list)
    ensemble_allowed: bool = True
    target_run_stopped: bool = False
    ensemble_branch_stopped: bool = False
    use_unweighted_states: bool = True
    use_crystal_baseline_only: bool = False
    confirmatory_generation_allowed: bool = False
    claims_allowed: set[str] = field(default_factory=set)
    withdrawn_claims: set[str] = field(default_factory=set)
    gate_log: list[GateResult] = field(default_factory=list)
    premarked_gates: dict[str, str] = field(default_factory=dict)
    exploratory_only: bool = True

