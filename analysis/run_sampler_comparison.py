"""Generate matched CMD-GEN DiffPhar and PharmFlow pharmacophore samples.

Uses the same receptor atom cloud, ligand-derived pocket center, feature count,
sample count, and seed for both samplers. This is a sampler pilot, not a complete
end-to-end drug-generation benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pathlib
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baseline_assets" / "extracted" / "CMD-GEN" / "code" / "CMD-GEN" / "DiffPhar"
sys.path.insert(0, str(ROOT))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pocket_from_files(receptor_path: Path, ligand_path: Path, center: np.ndarray,
                      radius: float, max_nodes: int, atom_encoder: dict,
                      device: torch.device) -> dict[str, torch.Tensor]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", str(receptor_path))[0]
    coordinates, atom_types = [], []
    for atom in structure.get_atoms():
        element = atom.element.strip().capitalize() if atom.element else atom.name[0]
        if element == "H" or np.linalg.norm(atom.coord - center) > radius:
            continue
        coordinates.append(atom.coord.astype(np.float32))
        atom_types.append(atom_encoder.get(element, atom_encoder.get("others", 10)))
    if not coordinates:
        raise ValueError(f"No receptor atoms found within {radius} A: {receptor_path}")
    coordinates = np.asarray(coordinates, dtype=np.float32)
    atom_types = np.asarray(atom_types, dtype=np.int64)
    nearest = np.argsort(np.linalg.norm(coordinates - center, axis=1))[:max_nodes]
    coordinates, atom_types = coordinates[nearest], atom_types[nearest]
    x = torch.as_tensor(coordinates, device=device)
    types = torch.as_tensor(atom_types, device=device, dtype=torch.long)
    one_hot = F.one_hot(types, num_classes=len(atom_encoder)).float()
    return {"x": x, "one_hot": one_hot,
            "size": torch.tensor([len(x)], device=device, dtype=torch.long),
            "mask": torch.zeros(len(x), device=device, dtype=torch.long)}


def ligand_center(ligand_path: Path) -> np.ndarray:
    if ligand_path.suffix.lower() == ".sdf":
        from rdkit import Chem
        molecules = Chem.SDMolSupplier(str(ligand_path), removeHs=True, sanitize=False)
        mol = next((m for m in molecules if m is not None), None)
        if mol is None or mol.GetNumConformers() == 0:
            raise ValueError(f"Could not read ligand coordinates: {ligand_path}")
        xyz = mol.GetConformer().GetPositions()
    else:
        model = PDBParser(QUIET=True).get_structure("ligand", str(ligand_path))[0]
        xyz = [a.coord for a in model.get_atoms()
               if a.element.strip().upper() not in {"H", "D"}]
    if len(xyz) == 0:
        raise ValueError(f"No heavy atoms in ligand reference: {ligand_path}")
    return np.asarray(xyz, dtype=np.float32).mean(axis=0)


def repeat_pocket(pocket: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
    x, h = pocket["x"], pocket["one_hot"]
    n = len(x)
    return {
        "x": x.repeat(batch_size, 1),
        "one_hot": h.repeat(batch_size, 1),
        "size": torch.full((batch_size,), n, device=x.device, dtype=torch.long),
        "mask": torch.repeat_interleave(torch.arange(batch_size, device=x.device), n),
    }


def load_baseline(checkpoint: Path, device: torch.device):
    # The published checkpoint was serialized on Linux and its repository uses
    # a Biopython helper removed in current Biopython. Map these at load time only.
    import Bio.PDB.Polypeptide as polypeptide
    polypeptide.three_to_one = lambda code: polypeptide.protein_letters_3to1_extended[code.upper()]
    pathlib.PosixPath = pathlib.WindowsPath
    sys.path.insert(0, str(BASELINE_ROOT))
    with contextlib.redirect_stdout(io.StringIO()):
        from lightning_modules import PharPocketDDPM
        model = PharPocketDDPM.load_from_checkpoint(str(checkpoint), map_location=device)
    model.to(device).eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("diffphar", "pharmflow"), required=True)
    parser.add_argument("--receptor", type=Path, required=True)
    parser.add_argument("--ligand", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--num-phars", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pocket-radius", type=float, default=10.0)
    parser.add_argument("--max-pocket-atoms", type=int, default=400)
    parser.add_argument("--diffphar-checkpoint", type=Path,
                        default=BASELINE_ROOT / "checkpoints" / "full_atom" / "best-model-epoch=epoch=281.ckpt")
    parser.add_argument("--pharmflow-checkpoint", type=Path,
                        default=ROOT / "checkpoints_pharmflow" / "best_model_epoch_46.ckpt")
    args = parser.parse_args()

    for p in (args.receptor, args.ligand):
        if not p.is_file():
            raise FileNotFoundError(p)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    center = ligand_center(args.ligand)

    if args.arm == "diffphar":
        model = load_baseline(args.diffphar_checkpoint, device)
        atom_encoder = model.pocket_type_encoder
        sampler, steps = model.ddpm, 500
    else:
        from step2_pharmflow import PharmFlowSampler
        wrapper = PharmFlowSampler(str(args.pharmflow_checkpoint), str(device), center,
                                   pocket_radius=args.pocket_radius)
        # The encoder vocabulary is identical to CMD-GEN's full-atom CrossDocked
        # vocabulary, including the fallback "others" class.
        from constants import dataset_params
        atom_encoder = dataset_params["crossdock_full"]["atom_encoder"]
        sampler, steps = wrapper.model, 50

    pocket = pocket_from_files(args.receptor, args.ligand, center,
                               args.pocket_radius, args.max_pocket_atoms,
                               atom_encoder, device)
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, args.n_samples, args.batch_size):
            # Reinitialize each matched batch so both samplers receive the same
            # initial random stream even though checkpoint loading consumes RNG.
            seed_all(args.seed + start)
            b = min(args.batch_size, args.n_samples - start)
            batch_pocket = repeat_pocket(pocket, b)
            nodes = torch.full((b,), args.num_phars, device=device, dtype=torch.long)
            if args.arm == "diffphar":
                from torch_scatter import scatter_mean
                xh, xh_pocket, phar_mask, pocket_mask = sampler.sample_given_pocket(
                    batch_pocket, nodes, timesteps=steps)
                # CMD-GEN's Lightning wrapper performs this frame restoration
                # after its low-level DDPM sampler returns centered coordinates.
                before = scatter_mean(batch_pocket["x"], pocket_mask, dim=0)
                after = scatter_mean(xh_pocket[:, :3], pocket_mask, dim=0)
                xh[:, :3] += (before - after)[phar_mask]
                xyz = xh[:, :3].reshape(b, args.num_phars, 3)
                classes = xh[:, 3:].argmax(dim=-1).reshape(b, args.num_phars)
            else:
                xyz, one_hot, mask, _ = sampler.sample_given_pocket(
                    batch_pocket, nodes, num_steps=steps, device=device)
                xyz = xyz.reshape(b, args.num_phars, 3)
                classes = one_hot.argmax(dim=-1).reshape(b, args.num_phars)
            rows.append((xyz.detach().cpu().numpy(), classes.detach().cpu().numpy()))

    coords = np.concatenate([x for x, _ in rows], axis=0)
    classes = np.concatenate([x for _, x in rows], axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    decoder = ["Aromatic", "Hydrophobe", "PosIonizable", "NegIonizable",
               "Acceptor", "Donor", "LumpedHydrophobe", "others"]
    np.savez_compressed(args.out, coords=coords, class_indices=classes,
                        class_names=np.asarray(decoder))
    metadata = {
        "arm": args.arm, "target": args.target, "seed": args.seed,
        "n_samples": args.n_samples, "num_pharmacophores": args.num_phars,
        "steps": steps, "device": str(device), "pocket_radius_angstrom": args.pocket_radius,
        "max_pocket_atoms": args.max_pocket_atoms, "pocket_atoms": int(pocket["size"].item()),
        "ligand_center_angstrom": center.tolist(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "checkpoint": str(args.diffphar_checkpoint if args.arm == "diffphar" else args.pharmflow_checkpoint),
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
