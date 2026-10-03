import argparse
from pathlib import Path
import json
import torch
import numpy as np

import utils
from lightning_modules import PharPocketDDPM
from equivariant_diffusion.dynamics import EGNNDynamics
from flow_matching import FlowMatching
from constants import dataset_params, FLOAT_TYPE, INT_TYPE


def load_fm_checkpoint(ckpt_path, device, dataset_name='crossdock', tanh=None):
    """Load standalone Flow Matching checkpoint (EGNNDynamics weights only)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Infer architecture from state dict
    sd = ckpt['model_state_dict']
    
    # Strip 'dynamics.' prefix
    dynamics_sd = {}
    for k, v in sd.items():
        if k.startswith('dynamics.'):
            dynamics_sd[k[9:]] = v  # remove 'dynamics.'
        else:
            dynamics_sd[k] = v
    
    # Infer dimensions
    phar_nf = 8  # 8 pharmacophore classes for crossdock
    residue_nf = 20  # 20 amino acids for CA representation
    joint_nf = dynamics_sd['phar_encoder.2.weight'].shape[0]  # output of phar_encoder
    hidden_nf = dynamics_sd['egnn.embedding.weight'].shape[0]
    in_node_nf = dynamics_sd['egnn.embedding.weight'].shape[1]
    n_layers = max([int(k.split('e_block_')[1].split('.')[0]) 
                    for k in dynamics_sd.keys() if 'e_block_' in k]) + 1
    attention = any('att_mlp' in k for k in dynamics_sd.keys())
    
    print(f'Inferred architecture: joint_nf={joint_nf}, hidden_nf={hidden_nf}, '
          f'n_layers={n_layers}, attention={attention}')
    print(f'phar_nf={phar_nf}, residue_nf={residue_nf}')
    
    # Build EGNNDynamics
    net_dynamics = EGNNDynamics(
        phar_nf=phar_nf,
        residue_nf=residue_nf,
        n_dims=3,
        joint_nf=joint_nf,
        device=device,
        hidden_nf=hidden_nf,
        act_fn=torch.nn.SiLU(),
        n_layers=n_layers,
        attention=attention,
        tanh=tanh if tanh is not None else False,
        norm_constant=1,
        inv_sublayers=1,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method='sum',
        edge_cutoff=6.0,
        update_pocket_coords=False,
    )
    
    # Load weights strictly
    missing, unexpected = net_dynamics.load_state_dict(dynamics_sd, strict=False)
    if missing:
        raise RuntimeError(f'Missing keys in checkpoint: {missing}')
    if unexpected:
        raise RuntimeError(f'Unexpected keys in checkpoint: {unexpected}')
    print('Checkpoint loaded successfully (all keys matched)')
    
    # Wrap in FlowMatching
    fm_module = FlowMatching(
        dynamics=net_dynamics,
        dataset_name=dataset_name,
        phar_nf=phar_nf,
        residue_nf=residue_nf,
        n_dims=3,
        lambda_h=1.0,
    )
    fm_module = fm_module.to(device)
    fm_module.eval()
    
    return fm_module


def is_lightning_checkpoint(ckpt_path):
    """Check if checkpoint is a Lightning checkpoint (has hyper_parameters)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return 'hyper_parameters' in ckpt


def build_pocket_from_pdb(pdb_file, resi_list, ref_ligand, dataset_name, device):
    """Build pocket dict from PDB file."""
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1 as three_to_one
    import torch.nn.functional as F
    
    pdb_struct = PDBParser(QUIET=True).get_structure('', pdb_file)[0]
    if resi_list is not None:
        residues = [
            pdb_struct[x.split(':')[0]][(' ', int(x.split(':')[1]), ' ')]
            for x in resi_list]
    else:
        residues = utils.get_pocket_from_ligand(pdb_struct, ref_ligand)

    pocket_coord = torch.tensor(np.array(
        [res['CA'].get_coord() for res in residues]),
        device=device, dtype=FLOAT_TYPE)
    pocket_types = torch.tensor(
        [dataset_params[dataset_name]['aa_encoder'][three_to_one[res.get_resname()]]
         for res in residues], device=device)

    pocket_one_hot = F.one_hot(
        pocket_types, num_classes=len(dataset_params[dataset_name]['aa_encoder'])
    )

    pocket_size = torch.tensor([len(pocket_coord)], device=device, dtype=INT_TYPE)
    pocket_mask = torch.zeros(len(pocket_coord), device=device, dtype=INT_TYPE)

    pocket = {
        'x': pocket_coord.unsqueeze(0),  # (1, M, 3)
        'one_hot': pocket_one_hot.unsqueeze(0),  # (1, M, 20)
        'size': pocket_size,  # (1,)
        'mask': pocket_mask  # (M,)
    }
    return pocket


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--pdbfile', type=str)
    parser.add_argument('--resi_list', type=str, nargs='+', default=None)
    parser.add_argument('--ref_ligand', type=str, default=None)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--num_nodes_phar', type=int, default=3)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--resamplings', type=int, default=10)
    parser.add_argument('--jump_length', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='crossdock',
                        choices=['crossdock', 'crossdock_full'],
                        help='Dataset name for Flow Matching checkpoint')
    parser.add_argument('--tanh', action='store_true',
                        help='Use tanh in EGNNDynamics EquivariantUpdate (matches training config)')
    args = parser.parse_args()

    pdb_id = Path(args.pdbfile).stem

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load model based on checkpoint type
    if is_lightning_checkpoint(args.checkpoint):
        print('Loading Lightning checkpoint...')
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model_type = ckpt.get('hyper_parameters', {}).get('model_type', 'ddpm')
        print(f'Model type: {model_type}')
        
        model = PharPocketDDPM.load_from_checkpoint(
            args.checkpoint, map_location=device)
        model = model.to(device)
        model.eval()
        
        if args.num_nodes_phar is not None:
            num_nodes_phar = torch.ones(args.n_samples, dtype=int) * \
                            args.num_nodes_phar
        else:
            num_nodes_phar = None

        if model_type == 'flow_matching':
            # Use FM generation path
            # Build pocket from PDB
            from Bio.PDB import PDBParser
            from Bio.PDB.Polypeptide import protein_letters_3to1 as three_to_one
            import torch.nn.functional as F
            from torch_scatter import scatter_mean
            from constants import FLOAT_TYPE, INT_TYPE
            
            pdb_struct = PDBParser(QUIET=True).get_structure('', args.pdbfile)[0]
            if args.resi_list is not None:
                residues = [
                    pdb_struct[x.split(':')[0]][(' ', int(x.split(':')[1]), ' ')]
                    for x in args.resi_list]
            else:
                residues = utils.get_pocket_from_ligand(pdb_struct, args.ref_ligand)

            pocket_coord = torch.tensor(np.array(
                [res['CA'].get_coord() for res in residues]),
                device=device, dtype=FLOAT_TYPE)
            pocket_types = torch.tensor(
                [dataset_params[args.dataset]['aa_encoder'][three_to_one[res.get_resname()]]
                 for res in residues], device=device)

            pocket_one_hot = F.one_hot(
                pocket_types, num_classes=len(dataset_params[args.dataset]['aa_encoder'])
            )

            pocket_size = torch.tensor([len(pocket_coord)] * args.n_samples,
                                       device=device, dtype=INT_TYPE)
            pocket_mask = torch.repeat_interleave(
                torch.arange(args.n_samples, device=device, dtype=INT_TYPE),
                len(pocket_coord)
            )

            pocket = {
                'x': pocket_coord.repeat(args.n_samples, 1),
                'one_hot': pocket_one_hot.repeat(args.n_samples, 1),
                'size': pocket_size,
                'mask': pocket_mask
            }

            if args.num_nodes_phar is not None:
                num_nodes_phar = torch.ones(args.n_samples, dtype=int, device=device) * \
                                args.num_nodes_phar
            else:
                num_nodes_phar = model.fm_module.size_distribution.sample_conditional(
                    n1=None, n2=pocket['size'])

            num_steps = args.timesteps if args.timesteps else 50
            x_final, h_onehot, phar_mask, _ = model.fm_module.sample_given_pocket(
                pocket, num_nodes_phar, num_steps=num_steps)

            # Build output dict - group by batch index (phar_mask)
            phar_mask = phar_mask.cpu()
            x = x_final.detach().cpu()
            phar_type = h_onehot.argmax(dim=1).detach().cpu()

            phar_to_coords = {}
            phar_decoder = dataset_params[args.dataset]['phar_decoder']
            
            # Group by molecule (batch index)
            for mol_idx in range(args.n_samples):
                mol_mask = (phar_mask == mol_idx)
                if not mol_mask.any():
                    continue
                mol_coords = x[mol_mask]
                mol_types = phar_type[mol_mask]
                
                molecule_name = f"Molecule_{mol_idx + 1}"
                phar_to_coords[molecule_name] = {}
                
                for atom_name_idx, coords in zip(mol_types, mol_coords):
                    atom_name = phar_decoder[atom_name_idx.item()]
                    if atom_name not in phar_to_coords[molecule_name]:
                        phar_to_coords[molecule_name][atom_name] = []
                    phar_to_coords[molecule_name][atom_name].append(coords.tolist())
        else:
            # DDPM path
            phar_to_coords = model.generate_phars(
                args.pdbfile, args.n_samples, args.resi_list, args.ref_ligand,
                num_nodes_phar, args.sanitize, largest_frag=not args.all_frags,
                relax_iter=(200 if args.relax else 0),
                resamplings=args.resamplings, jump_length=args.jump_length,
                timesteps=args.timesteps)
    else:
        print('Loading standalone Flow Matching checkpoint...')
        fm_module = load_fm_checkpoint(args.checkpoint, device, args.dataset, tanh=args.tanh)
        
        # Build pocket from PDB
        pocket = build_pocket_from_pdb(args.pdbfile, args.resi_list, args.ref_ligand, args.dataset, device)
        
        # Replicate pocket for n_samples
        M = pocket['x'].shape[1]
        pocket = {
            'x': pocket['x'].repeat(args.n_samples, 1, 1).view(-1, 3),  # (n_samples * M, 3)
            'one_hot': pocket['one_hot'].repeat(args.n_samples, 1, 1).view(-1, pocket['one_hot'].shape[-1]),  # (n_samples * M, 20)
            'size': torch.full((args.n_samples,), M, device=device, dtype=INT_TYPE),
            'mask': torch.repeat_interleave(torch.arange(args.n_samples, device=device, dtype=INT_TYPE), M)
        }

        if args.num_nodes_phar is not None:
            num_nodes_phar = torch.ones(args.n_samples, dtype=int, device=device) * \
                            args.num_nodes_phar
        else:
            num_nodes_phar = fm_module.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])

        num_steps = args.timesteps if args.timesteps else 50
        x_final, h_onehot, phar_mask, _ = fm_module.sample_given_pocket(
            pocket, num_nodes_phar, num_steps=num_steps)

        # Build output dict - group by batch index (phar_mask)
        phar_mask = phar_mask.cpu()
        x = x_final.detach().cpu()
        phar_type = h_onehot.argmax(dim=1).detach().cpu()

        phar_to_coords = {}
        phar_decoder = dataset_params[args.dataset]['phar_decoder']
        
        # Group by molecule (batch index)
        for mol_idx in range(args.n_samples):
            mol_mask = (phar_mask == mol_idx)
            if not mol_mask.any():
                continue
            mol_coords = x[mol_mask]
            mol_types = phar_type[mol_mask]
            
            molecule_name = f"Molecule_{mol_idx + 1}"
            phar_to_coords[molecule_name] = {}
            
            for atom_name_idx, coords in zip(mol_types, mol_coords):
                atom_name = phar_decoder[atom_name_idx.item()]
                if atom_name not in phar_to_coords[molecule_name]:
                    phar_to_coords[molecule_name][atom_name] = []
                phar_to_coords[molecule_name][atom_name].append(coords.tolist())

    # Save output
    output_file_path = Path(f'phar_to_coords_{pdb_id}.json')
    with open(output_file_path, 'w') as f:
        json.dump(phar_to_coords, f, indent=2)

    print(f'Saved to {output_file_path}')
    print(json.dumps(phar_to_coords, indent=2))
