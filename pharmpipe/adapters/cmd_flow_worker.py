"""Subprocess entry point for inference in the CMD-fork environment."""

from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath
if "pathlib._local" not in sys.modules:
    sys.modules["pathlib._local"] = pathlib


def _to_json_serializable(val):
    if hasattr(val, "tolist"):
        return val.tolist()
    if isinstance(val, dict):
        return {k: _to_json_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_json_serializable(v) for v in val]
    return val


def main() -> None:
    request_path, output_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    repository = Path(request["repository_dir"]).resolve()
    sys.path.insert(0, str(repository))

    import numpy as np
    import torch
    from lightning_modules import PharPocketDDPM

    seed = int(request["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PharPocketDDPM.load_from_checkpoint(
        request["checkpoint_path"], map_location=device
    )
    model.eval()
    if getattr(model, "model_type", None) != "flow_matching":
        raise ValueError("Checkpoint model_type is not flow_matching")

    predictions = {}
    for state in request["states"]:
        state_samples = {}
        for sample_index in range(int(request["samples_per_state"])):
            sample_seed = seed + sample_index
            torch.manual_seed(sample_seed)
            np.random.seed(sample_seed)
            generated = model.generate_phars(
                state["pdb_path"],
                1,
                pocket_ids=state["pocket_residues"],
                ref_ligand=None,
                num_nodes_phar=None,
                sanitize=False,
                largest_frag=False,
                relax_iter=0,
                timesteps=int(request["timesteps"]),
            )
            # CMD-fork returns a dictionary keyed by sample name, with feature
            # labels mapped to coordinate lists.
            if isinstance(generated, dict) and generated and all(
                isinstance(value, dict) for value in generated.values()
            ):
                state_samples.update(_to_json_serializable(generated))
            elif isinstance(generated, dict):
                state_samples[f"sample_{sample_index:04d}"] = _to_json_serializable(generated)
            else:
                raise TypeError("generate_phars returned an unsupported result type")
        predictions[state["state_id"]] = state_samples

    output_path.write_text(
        json.dumps({"states": predictions}), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
