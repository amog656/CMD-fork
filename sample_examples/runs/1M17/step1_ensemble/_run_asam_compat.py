import runpy
import sys
import types
import torch
if not hasattr(torch, 'accelerator'):
    # Diffusers 0.41 queries this API at import time; a CPU fallback
    # keeps it compatible with the project's Torch 2.1 environment.
    torch.accelerator = types.SimpleNamespace(is_available=lambda: False)
script = 'C:\\Users\\hanut sai\\Desktop\\nnn\\asam_engine\\src\\sam\\scripts\\generate_ensemble.py'
sys.argv = [script, *sys.argv[1:]]
runpy.run_path(script, run_name='__main__')
