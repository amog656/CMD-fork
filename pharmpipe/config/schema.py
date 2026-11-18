"""Validated Target Configuration Card schema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConstructConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chain: str
    residues: tuple[int, int]

    @model_validator(mode="after")
    def valid_residue_range(self):
        if self.residues[0] > self.residues[1]:
            raise ValueError("construct.residues must be an inclusive ascending range")
        return self


class TargetConfigBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    name: str
    construct_info: ConstructConfig = Field(alias="construct")
    structure_path: str | None = None
    pocket_source: Literal["residues", "detector"]
    pocket_residues: list[str | int] = Field(default_factory=list)


class OffTargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    structure_path: str


class ReferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actives_path: str | None = None
    inactives_path: str | None = None
    decoys_path: str | None = None


class PropertyGates(BaseModel):
    model_config = ConfigDict(extra="forbid")
    MW: float | None = 400
    LogP: float | None = 3
    QED: float | None = 0.6
    SAS: float | None = 4
    rotatable_bonds: int | None = None
    TPSA: float | None = None


class NMinima(BaseModel):
    model_config = ConfigDict(extra="forbid")
    G1_reference_states: int = 1
    G3_labelled_units: int = 30
    G2b_actives: int = 100
    G2b_decoys: int = 100
    G4_actives: int = 100
    G4_decoys: int = 100
    G4_EF1_actives: int = 100
    G4_EF1_decoys: int = 9900


class MergeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vote_fraction: float = Field(default=0.5, ge=0, le=1)
    group_distance_A: float = Field(default=1.5, gt=0)


class TargetConfig(BaseModel):
    """TCC with defaults kept here and annotated in resolved provenance."""

    model_config = ConfigDict(extra="forbid")
    target: TargetConfigBlock
    off_targets: list[OffTargetConfig] = Field(default_factory=list)
    mode: Literal["A", "B", "C", "D"]
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    selectivity_goal_fold: float = Field(default=10, gt=1)
    property_gates: PropertyGates = Field(default_factory=PropertyGates)
    state_weight_policy: Literal["unweighted", "reference_T"] = "unweighted"
    K_states: int = Field(default=5, gt=0)
    fep_shortlist_size: int = Field(default=15, gt=0)
    n_minima: NMinima = Field(default_factory=NMinima)
    merge: MergeConfig = Field(default_factory=MergeConfig)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0

    @model_validator(mode="after")
    def mode_matches_off_targets(self):
        count = len(self.off_targets)
        expected = "A" if count == 0 else "B" if count == 1 else "C"
        if self.mode != "D" and self.mode != expected:
            raise ValueError(
                f"mode {self.mode!r} disagrees with {count} off-target(s); expected {expected!r} or 'D'"
            )
        return self
