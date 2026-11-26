"""
Class for streamline usage of the SAM minimizer.
"""

import os
import yaml
import numpy as np
import mdtraj
import torch

from sam.data.aa_topology import sam_openfold_aa_map, get_traj_list
from sam.minimizer import get_topology, initialize, minimize, reconstruct_atom14
from sam.data.aa_protein import AllAtomProteinDataset


class Minimizer:

    def __init__(self,
            name: str,
            top_fp: str,
            ens_fp: str,
            protocol: str,
            params_fp: str = None
        ):
        
        # Load some data.
        self.dataset = AllAtomProteinDataset(
            input=[{
                "name": name, "topology": top_fp, "trajectories": [ens_fp]
            }],
            n_trajs=None,
            n_frames=None,
            frames_mode="ensemble",
            proteins=None,
            per_protein_frames=None,
            re_filter=None,
            res_ids_mode=None,
            bead_type="ca",
            alphabet="standard",
            xyz_sigma=None,
            xyz_perturb=None,
            verbose=False,
            random_seed=None
        )

        # Load the minimization parameters.
        module_dp = os.path.dirname(__file__)
        if protocol == "atlas":
            params_fp = os.path.join(module_dp, "params", "mizu_cfg.atlas.yaml")
        elif protocol == "mdcath":
            params_fp = os.path.join(module_dp, "params", "mizu_cfg.mdcath.yaml")
        elif protocol == "custom":
            if params_fp is None:
                raise ValueError()
        else:
            raise KeyError(protocol)
        with open(params_fp, 'r') as i_fh:
            params = yaml.safe_load(i_fh)
        self.opt_params = params["opt"]
        if "opt_ini" in params:
            self.opt_ini_params = params["opt_ini"]
        else:
            self.opt_ini_params = None
        self.top_params = params["top"]
        self.data_params = params.get("data", {"batch_size": 50})


    def run(self,
            batch_size: int = None,
            device: str = "cpu",
            verbose: bool = True
        ):
        # Setup the batch size.
        if batch_size is None:
            batch_size = self.data_params["batch_size"]

        # Setup the dataloader to serve the batches to minimize.
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=self.dataset, batch_size=batch_size, shuffle=False
        )

        # Iterate over the whole dataset in batches.
        minimized_traj = []
        for i, batch in enumerate(dataloader):

            # Get the xyz coordinates and the amino acid sequence.
            positions = batch.atom14_gt_positions
            a = torch.tensor(sam_openfold_aa_map[batch.a], dtype=torch.long)
            positions = positions*0.1
            positions = positions.to(device)
            a = a.to(device)

            # Initialize.
            if i == 0:
                topology = get_topology(a, **self.top_params)
            positions = initialize(positions, a)

            # Brief initial minimization, typically with simple GD or Adam.
            if self.opt_ini_params is not None:
                positions, es = minimize(
                    positions=positions,
                    topology=topology,
                    return_early_stopping=True,
                    verbose=verbose,
                    **self.opt_ini_params
                )
                if not es:
                    positions = torch.autograd.Variable(positions)
                    positions.requires_grad = True
            else:
                es = False

            # Main minimization, typically with L-BFGS.
            if not es:
                # Run only if did not early-stop at the initial minimization.
                positions = minimize(
                    positions=positions,
                    topology=topology,
                    verbose=verbose,
                    **self.opt_params
                )

            # Reconstruct atom14 trajectory.
            atom14_rec = reconstruct_atom14(positions, topology)
            traj = get_traj_list({"positions": atom14_rec, "a": a}, join=True)
            minimized_traj.append(traj)

        # Join the mdtraj trajectories and return.
        minimized_traj = mdtraj.join(minimized_traj)

        return minimized_traj