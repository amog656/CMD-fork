"""
Compare two ensembles of the same proteins using the ensemble comparison
and analysis scores used in the aSAM article (see Table 1). Specifically:
    (i) PCC Ca RMSF,
    (ii) chiJSD,
    (iii) heavy clashes,
    (iv) peptide bond length violations.
"""

import os
import json
import pathlib
import argparse
import numpy as np
import mdtraj
from sam.data.topology import slice_ca_traj
from sam.evaluation.scores import score_pcc_ca_rmsf, score_chiJSD, mstats_stereo

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-P', '--ref_top', type=str, required=True,
        help='topology file of the reference ensemble'
    )
    parser.add_argument('-T', '--ref_traj', type=str, required=True,
        help='trajectory file of the reference ensemble'
    )
    parser.add_argument('-p', '--hat_top', type=str, required=True,
        help='topology file of the proposed ensemble'
    )
    parser.add_argument('-t', '--hat_traj', type=str, required=True,
        help='trajectory file of the proposed ensemble'
    )
    parser.add_argument('-i', '--init_pdb', type=str, required=True,
        help='PDB file of some initial (or reference) structure'
    )
    args = parser.parse_args()

    # Will store the results here.
    json_data = {}

    # Load the heavy atom (ha) ensembles.
    ref_ha_traj = mdtraj.load(args.ref_traj, top=args.ref_top)
    hat_ha_traj = mdtraj.load(args.hat_traj, top=args.hat_top)
    ini_ha_traj = mdtraj.load(args.init_pdb)
    # Create the Ca ensembles.
    ref_ca_traj = slice_ca_traj(ref_ha_traj)
    hat_ca_traj = slice_ca_traj(hat_ha_traj)
    ini_ca_traj = slice_ca_traj(ini_ha_traj)

    # Score PCC Ca RMSF.
    pcc_ca_rmsf = score_pcc_ca_rmsf(
        ref_ca_traj=ref_ca_traj,
        hat_ca_traj=hat_ca_traj,
        ini_ca_traj=ini_ca_traj
    )
    json_data["pcc_ca_rmsf"] = float(pcc_ca_rmsf)

    # Score chiJSD.
    chi_jsd = score_chiJSD(
        ref_traj=ref_ha_traj, hat_traj=hat_ha_traj
    )
    json_data["chi_jsd"] = float(chi_jsd)

    # Heavy clashes and peptide bond length violations.
    stats = mstats_stereo(ref_ha_traj)
    json_data[f"ref_heavy_clash"] = float(np.mean(stats["heavy_clash_ha"]))
    json_data[f"ref_viol_c_n"] = float(np.mean(stats["viol_c_n"]))
    stats = mstats_stereo(hat_ha_traj)
    json_data[f"hat_heavy_clash"] = float(np.mean(stats["heavy_clash_ha"]))
    json_data[f"hat_viol_c_n"] = float(np.mean(stats["viol_c_n"]))

    # Print results.
    print(json_data)

if __name__ == "__main__":

    main()
