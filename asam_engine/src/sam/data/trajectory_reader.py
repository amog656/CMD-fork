"""
Load trajectory data.
"""
import numpy as np
import mdtraj
from sam.data.topology import slice_ca_traj
from sam.coords import sample_data

def get_ca_traj(
    input: dict,
    slice_traj: bool = True,
    n_frames: int = 1000,
    n_trajs: int = None,
    frames_mode: str = "ensemble",
    get_xyx: bool = False):

    # Read xyz data from a trajectory file.
    xyz = []
    if slice_traj:
        slice_func = slice_ca_traj
    else:
        slice_func = lambda t: t
    top_traj = slice_func(mdtraj.load(input["topology"]))

    if n_trajs is None:
        sel_trajectories = input["trajectories"]
    else:
        sel_trajectories = np.random.choice(input["trajectories"], n_trajs,
                                            replace=False)

    # Actually parse each trajectory file.
    for traj_fp_i in sel_trajectories:
        # Load the trajectory.
        traj_i = slice_func(
            mdtraj.load(traj_fp_i, top=input["topology"]))
        xyz_i = traj_i.xyz
        # Sample frames with mode "trajectory".
        if frames_mode == "trajectory":
            xyz_i = sample_data(data=xyz_i, n_samples=n_frames)
        if xyz_i.shape[0] == 0:
            raise ValueError()
        # Store the frames.
        xyz.append(xyz_i)

    # Begin preparing the results to return.
    if not xyz:
        raise ValueError("No data found in: {}".format(
            repr(input["trajectories"])))
    xyz = np.concatenate(xyz, axis=0)
    
    # Sample frames with mode "ensemble".
    if frames_mode == "ensemble":
        xyz = sample_data(data=xyz, n_samples=n_frames)
    if get_xyx:
        return xyz
    else:
        traj = mdtraj.Trajectory(xyz=xyz, topology=top_traj.topology)
        return traj