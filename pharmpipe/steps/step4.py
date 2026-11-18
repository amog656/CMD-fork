"""Step 4: mock PharmFlow inference and non-evaluable model validation gates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..adapters.pharmflow import MockPharmFlowAdapter
from ..config.loader import LoadedConfig
from ..data.models import GateResult, PharmModel, PocketState
from ..gates.g2a import evaluate_g2a
from ..gates.glfo import evaluate_glfo
from ..runio import prepare_stage_dir, write_manifest


@dataclass
class Step4Result:
    models: list[PharmModel]
    g_lfo: GateResult
    g2a: GateResult
    models_path: Path


def run_step4(loaded: LoadedConfig, *, states: list[PocketState], output_dir: str | Path, minimum_train_families: int, minimum_held_out_families: int, adapter=None) -> Step4Result:
    out = Path(output_dir)
    prepare_stage_dir(out)
    model_adapter = adapter or MockPharmFlowAdapter()
    accepted = [state for state in states if state.in_domain]
    if hasattr(model_adapter, "predict_many"):
        models = model_adapter.predict_many(accepted, loaded.config.seed)
    else:
        models = [model_adapter.predict(state, loaded.config.seed + index) for index, state in enumerate(accepted)]
    records = [{"model_id": model.model_id, "origin": model.origin, "supporting_state_ids": model.supporting_state_ids, "vote_counts": model.vote_counts, "points": [asdict(point) for point in model.points]} for model in models]
    path = out / "pharmacophores.json"
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trained_checkpoint = not isinstance(model_adapter, MockPharmFlowAdapter)
    write_manifest(out, config_hash=loaded.config_hash, seed=loaded.config.seed, counts_in={"states": len(states), "in_domain_states": len(accepted)}, counts_out={"models": len(models)}, dropped=[{"state_id": state.state_id, "reason": "out_of_domain"} for state in states if not state.in_domain], extra={"adapter": getattr(model_adapter, "version", type(model_adapter).__name__), "checkpoint_inference": trained_checkpoint, "point_pooling": "per-feature Gaussian mixture selected by BIC" if trained_checkpoint else "mock"})
    return Step4Result(models, evaluate_glfo(trained=trained_checkpoint, train_families=0, held_out_families=0, minimum_train_families=minimum_train_families, minimum_held_out_families=minimum_held_out_families, config_hash=loaded.config_hash), evaluate_g2a(trained=trained_checkpoint, config_hash=loaded.config_hash), path)
