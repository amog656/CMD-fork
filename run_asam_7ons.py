import os
import sys
import time
from pathlib import Path

# Memory optimization for 4GB VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from sam.scripts.generate_ensemble import generate_ensemble

def main():
    print("=" * 70, flush=True)
    print("  GANDIVA: aSAM Conformational Sampling (RTX 3050 Optimized)", flush=True)
    print("=" * 70, flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    
    input_pdb = Path("7ONS_chainA.pdb").resolve()
    out_prefix = Path("ensembles/7ONS_ensemble/conf").resolve()
    out_dir = out_prefix.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 20 states, batch_size=1, 25 diffusion steps for fast high-quality convergence
    print(f"Target: {input_pdb.name}", flush=True)
    print(f"Generating: 20 conformations via aSAM latent diffusion...", flush=True)
    print(f"Settings: batch_size=1 (4GB VRAM guard), n_steps=25, no_minimize=True", flush=True)
    
    t0 = time.time()
    generate_ensemble(
        init=input_pdb,
        out_path=out_prefix,
        n_samples=20,
        batch_size=1,
        n_steps=25,
        device="cuda",
        no_minimize=True,
        track_time=True,
    )
    elapsed = time.time() - t0
    print(f"\nSUCCESS: aSAM generated 20 states in {elapsed:.1f}s ({elapsed/60:.2f} mins)!", flush=True)
    
    # Split trajectory into individual conf_0001.pdb ... conf_0020.pdb for pharmpipe
    import mdtraj as md
    traj_dcd = out_dir / "conf.dcd"
    top_pdb = out_dir / "conf.pdb"
    if traj_dcd.exists() and top_pdb.exists():
        print("Extracting individual PDB frames for PharmPipe ensemble ingestion...", flush=True)
        traj = md.load(str(traj_dcd), top=str(top_pdb))
        for i in range(len(traj)):
            frame_path = out_dir / f"conf_{i+1:04d}.pdb"
            traj[i].save_pdb(str(frame_path))
        print(f"Saved {len(traj)} individual frame PDBs (conf_0001.pdb to conf_{len(traj):04d}.pdb).", flush=True)
    
    # Write completion marker for pipeline
    marker = out_dir / ".done"
    marker.write_text("COMPLETED_20_FRAMES", encoding="utf-8")
    print(f"Created marker: {marker} -> COMPLETED_20_FRAMES")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
