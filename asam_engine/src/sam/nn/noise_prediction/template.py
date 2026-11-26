import torch
import torch.nn as nn
from sam.coords import (
    calc_torsion, calc_chain_bond_angles, torch_chain_dihedrals, calc_com_traj
)
from sam.nn.common import get_act_fn
from sam.nn.geometric import BeadDistanceEmbedding, featurize_angle


class TemplateEdgeEmbedder(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
            edge_dim: int,
            dmap_embed_params: dict,
            no_dmap_embed_params: dict = {},
            com_dmap_embed_params: dict = {},
            activation: str = "relu",
            dmap_merge_mode: str = "cat_shallow",
            dmap_merge_dim: int = 128,  # NOTE: here.
            pair_update_net_params: dict = {},
            linear_bias: bool = True
        ):
        """Arguments: TODO."""
        super().__init__()

        if pair_update_net_params:
            raise NotImplementedError()

        # Embed Ca-Ca distances.
        self.dmap_embedder = BeadDistanceEmbedding(
            edge_dim=None,
            embed_type=dmap_embed_params["type"],
            activation=activation,
            params=dmap_embed_params,
            use_project=False,
            use_mlp=False,
            input_type="xyz"
        )
        dmap_merge_in = self.dmap_embedder.dmap_dim

        #+++++++++++++++++++
        # Embed N-O distances.
        if no_dmap_embed_params:
            self.no_dmap_embedder = BeadDistanceEmbedding(
                edge_dim=None,
                embed_type=no_dmap_embed_params["type"],
                activation=activation,
                params=no_dmap_embed_params,
                use_project=False,
                use_mlp=False,
                input_type="dmap"
            )
        else:
            self.no_dmap_embedder = None
        
        # Embed COM-COM distances.
        if com_dmap_embed_params:
            self.com_dmap_embedder = BeadDistanceEmbedding(
                edge_dim=None,
                embed_type=com_dmap_embed_params["type"],
                activation=activation,
                params=com_dmap_embed_params,
                use_project=False,
                use_mlp=False,
                input_type="xyz"
            )
        else:
            self.com_dmap_embedder = None
        
        # Module to merge the distance maps of various beads.
        dmap_modules = (self.no_dmap_embedder, self.com_dmap_embedder)
        # if self.no_dmap_embedder is not None or self.com_dmap_embedder is not None:
        if any(mod is not None for mod in dmap_modules):
            if dmap_merge_mode == "cat_shallow":
                # Concatenate.
                if self.no_dmap_embedder is not None:
                    dmap_merge_in += self.no_dmap_embedder.dmap_dim
                if self.com_dmap_embedder is not None:
                    dmap_merge_in += self.com_dmap_embedder.dmap_dim
            else:
                raise KeyError(dmap_merge_mode)
            # dmap_exp_dim = dmap_merge_dim
            self.dmap_merge_mode = dmap_merge_mode

        else:
            # dmap_exp_dim = self.dmap_embedder.dmap_dim
            self.dmap_merge_mode = None
        #+++++++++++++++++++

        self.dmap_merger = nn.Sequential(
            nn.Linear(dmap_merge_in, dmap_merge_dim, bias=linear_bias),
            get_act_fn(activation)(),
            nn.Linear(dmap_merge_dim, edge_dim, bias=linear_bias),
        )

    def forward(self, x, top):
        p_ca = self.dmap_embedder(x[:,:,1,:])
        #+++++++++++++++++++
        # Get other distance matrices.
        if self.dmap_merge_mode is not None:
            # N-O distance matrix.
            if self.no_dmap_embedder is not None:
                dmap_no = self.calc_dmap(x[:,:,0,:], x[:,:,3,:])
                i = torch.arange(0, dmap_no.shape[2])
                dmap_no_exp = self.no_dmap_embedder(dmap_no)
            # COM-COM distance matrix.
            if self.com_dmap_embedder is not None:
                com = calc_com_traj(positions=x, atom14_gt_exists=top, mult=1.0)
                dmap_com_exp = self.com_dmap_embedder(com)
            # Embed via concatenation.
            if self.dmap_merge_mode == "cat_shallow":
                dmap_merger_input = [p_ca]
                if self.no_dmap_embedder is not None:
                    dmap_merger_input.append(dmap_no_exp)
                if self.com_dmap_embedder is not None:
                    dmap_merger_input.append(dmap_com_exp)
                dmap_merger_input = torch.cat(dmap_merger_input, dim=3)
            else:
                raise KeyError(self.dmap_merge_mode)
        else:
            dmap_merger_input = p_ca
        #+++++++++++++++++++
        p_e = self.dmap_merger(dmap_merger_input)
        return p_e

    def calc_dmap(self, x_1, x_2, epsilon=1e-12):
        dmap = torch.cdist(x_1, x_2, p=2.0)
        dmap = dmap.unsqueeze(1)
        return dmap


class TemplateNodeEmbedder(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
            node_dim: int,
            mode: str,
            angle_bins: int = 16,
            activation: str = "relu",
            mask_class: str = "empty",
            mlp_mult: int = 1,
            mlp_depth: int = 3
        ):
        """Arguments: TODO."""
        super().__init__()
        # Initialize.
        self.mode = mode
        self.angle_bins = angle_bins
        self.mask_class = mask_class

        if self.mask_class == "empty":
            mask_cols = 0
        elif self.mask_class == "extra":
            mask_cols = 1
        else:
            raise KeyError(self.mask_class)

        # MLP.
        num_feats = (self.angle_bins+1)*5
        act_cls = get_act_fn(activation)
        mlp_modules = [
            nn.Linear(num_feats, node_dim*mlp_mult),
            act_cls()
        ]
        if mlp_depth == 2:
            pass
        elif mlp_depth == 3:
            mlp_modules.extend([
                nn.Linear(node_dim*mlp_mult, node_dim*mlp_mult),
                act_cls()
            ])
        else:
            raise ValueError(mlp_depth)
        mlp_modules.append(nn.Linear(node_dim*mlp_mult, node_dim))
        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, x, top, epsilon=1e-9):

        node_f = []
        ca_xyz = x[:,:,1,:]

        ## Alpha angles.
        alpha = torch_chain_dihedrals(ca_xyz)
        alpha_f = featurize_angle(
            alpha, (1, 2),
            bins=self.angle_bins, mask_class=self.mask_class
        )
        node_f.append(alpha_f)

        ## Beta angles.
        beta = calc_chain_bond_angles(ca_xyz, backend="torch")
        beta_f = featurize_angle(
            beta, (0, 2),
            bins=self.angle_bins, mask_class=self.mask_class
        )
        node_f.append(beta_f)

        ## Bond lengths.
        # bond_length = torch.sqrt(
        #     torch.square(ca_xyz[:,:-1,:] - ca_xyz[:,1:,:]).sum(dim=2) + epsilon
        # )
        # bond_length = bond_length.unsqueeze(2)
        # bond_length_f = torch.cat(
        #     [bond_length, torch.ones_like(bond_length)],
        #     dim=2
        # )
        # bond_length_f = nn.functional.pad(bond_length_f, (0, 0, 0, 1))
        # node_f.append(bond_length_f)
        
        if self.mode == "aa":

            # ## Chi angles.
            # chi_f = torch.cat(
            #     [batch.chi_mask.unsqueeze(3), batch.chi_angles_sin_cos],
            #     dim=3
            # )
            # chi_f = chi_f.view(chi_f.shape[0], chi_f.shape[1], -1)
            # node_f.append(chi_f)

            ## Phi/psi/omega angles.
            # PHI_ATOMS = ["-C", "N", "CA", "C"]
            phi = calc_torsion(x[:,0:-1,2,:],
                               x[:,1:,  0,:],
                               x[:,1:,  1,:],
                               x[:,1:,  2,:])
            phi_f = featurize_angle(
                phi, (1, 0),
                bins=self.angle_bins, mask_class=self.mask_class
            )
            node_f.append(phi_f)

            # PSI_ATOMS = ["N", "CA", "C", "+N"]
            psi = calc_torsion(x[:,0:-1,0,:],
                               x[:,0:-1,1,:],
                               x[:,0:-1,2,:],
                               x[:,1:,  0,:])
            psi_f = featurize_angle(
                psi, (0, 1),
                bins=self.angle_bins, mask_class=self.mask_class
            )
            node_f.append(psi_f)

            # OMEGA_ATOMS = ["CA", "C", "+N", "+CA"]
            omega = calc_torsion(x[:,0:-1,1,:],
                                 x[:,0:-1,2,:],
                                 x[:,1:,  0,:],
                                 x[:,1:,  1,:])
            omega_f = featurize_angle(
                omega, (0, 1),
                bins=self.angle_bins, mask_class=self.mask_class
            )
            node_f.append(omega_f)

        # ## Label N- and C-term.
        # if self.label_terminus:
        #     term_label = torch.zeros(
        #         batch.x.shape[0],
        #         batch.x.shape[1],
        #         device=batch.x.device
        #     )
        #     term_label[:,0] = 1
        #     term_label[:,-1] = 1
        #     node_f.append(term_label.unsqueeze(-1))
            
        ## Concatenate the features.
        node_f = torch.cat(node_f, dim=2)

        h_tem = self.mlp(node_f)

        return h_tem