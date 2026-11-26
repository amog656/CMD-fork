"""
Analyze an ensemble againts a reference structure and compute scores used in the
aSAM article. Specifically:
    (i) Folded state fraction (FSF),
    (ii) Secondary structure elements preservation (SSEP),
    (iii) average initRMSD (RMSD with respect to the reference structure)
"""

import os
import json
import pathlib
import argparse
import json
import mdtraj
from sam.trajectory import calc_initrmsd, calc_ssep, calc_q_values

def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', '--native_pdb', type=str, required=True,
        help='a reference PDB file, which may represent a native structure or'
             ' an initial structure for MD'
    )
    parser.add_argument('-p', '--ensemble_top', type=str, required=True,
        help='topology file of the input ensemble'
    )
    parser.add_argument('-t', '--ensemble_traj', type=str, required=True,
        help='trajectory file of the input ensemble'
    )
    parser.add_argument('--q_thresh', type=float, default=0.6,
        help='Q value threshold for defining the folded state'
    )
    args = parser.parse_args()

    # Will store the results here.
    json_data = {}

    # Load the ensemble and the native structure
    traj = mdtraj.load(args.ensemble_traj, top=args.ensemble_top)
    native_traj = mdtraj.load(args.native_pdb)

    # Compute FSF.
    # First compute Q values (we also use them in the paper and show their
    # histograms).
    q_values = calc_q_values(
        traj=traj,
        native_traj=native_traj,
        beta=50.0,
        lambda_=1.2,
        delta=0.0,
        threshold=1.0
    )
    # Assign folded/unfolded states.
    folded_state = q_values > args.q_thresh
    # Calculate the fraction of snapshots in the folded state.
    json_data["fsf"] = float(folded_state.mean())

    # Compute SSEP.
    seep = calc_ssep(traj=traj, native_traj=native_traj)
    json_data["seep"] = float(seep.mean())

    # Compute initRMSD.
    initrmsd = calc_initrmsd(
        traj=traj, init_traj=native_traj, is_ca=False, get_tm=False
    )
    json_data["avg_initrmsd"] = float(initrmsd.mean())

    # Print results.
    print(json_data)

if __name__ == "__main__":
    main()
