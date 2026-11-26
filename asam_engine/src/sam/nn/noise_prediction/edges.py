import torch
import torch.nn as nn
from sam.nn.common import get_act_fn
from sam.nn.af3 import Transition
from sam.nn.af2 import EdgeUpdaterESMFold


class EdgeUpdaterFrameDiff(nn.Module):
    """Edge representation updater from FrameDiff."""

    def __init__(self, node_dim, edge_dim, outer_operation="concat",
                 activation=nn.ReLU, use_in_ln=False, use_out_ln=True):
        super().__init__()
        self.down_linear = nn.Linear(node_dim, node_dim // 2)
        if outer_operation == "concat":
            edge_mlp_dim = node_dim + edge_dim  # (embed_dim // 2) * 2 + edge_dim
        elif outer_operation == "sum":
            edge_mlp_dim = node_dim // 2 + edge_dim
            # self.post_add_norm = nn.LayerNorm(node_dim // 2)
        else:
            raise KeyError(outer_operation)
        self.outer_operation = outer_operation
        
        if use_in_ln:
            self.edge_in_norm = nn.LayerNorm(edge_dim)
        self.use_in_ln = use_in_ln

        act_cls = get_act_fn(activation)
        self.edge_mlp = nn.Sequential(nn.Linear(edge_mlp_dim, edge_mlp_dim),
                                      act_cls(),
                                      nn.Linear(edge_mlp_dim, edge_mlp_dim),
                                      act_cls())

        self.edge_out_linear = nn.Linear(edge_mlp_dim, edge_dim)

        if use_out_ln:
            self.edge_out_norm = nn.LayerNorm(edge_dim)
        self.use_out_ln = use_out_ln

    def forward(self, x, z):
        ### x_down = self.down_linear(x).transpose(0, 1)
        x_down = self.down_linear(x)
        num_res = x_down.shape[1]
        if self.outer_operation == "concat":
            # x_down = x_down.unsqueeze(2).repeat(1, 1, x_down.shape[1], 1)
            # z_in = torch.cat([x_down, x_down.transpose(1, 2), z], dim=3)
            edge_bias = torch.cat([
                torch.tile(x_down[:, :, None, :], (1, 1, num_res, 1)),
                torch.tile(x_down[:, None, :, :], (1, num_res, 1, 1)),
            ], axis=-1)
        elif self.outer_operation == "sum":
            edge_bias = torch.tile(x_down[:, :, None, :], (1, 1, num_res, 1)) + \
                        torch.tile(x_down[:, None, :, :], (1, num_res, 1, 1))
        else:
            raise KeyError(self.outer_operation)
        if self.use_in_ln:
            z = self.edge_in_norm(z)
        z_in = torch.cat([edge_bias, z], axis=-1)
        z = self.edge_out_linear(self.edge_mlp(z_in) + z_in)
        if self.use_out_ln:
            z = self.edge_out_norm(z)
        return z


class EdgeUpdaterSAM_0(nn.Module):
    """Custom edge representation updater 0."""

    def __init__(self,
                 node_dim, edge_dim,
                 edge_downsample=2,
                 activation=nn.ReLU):
        super().__init__()
        hidden_dim = edge_dim // edge_downsample
        self.node_input_linear = nn.Linear(node_dim, hidden_dim)
        self.edge_input_linear = nn.Linear(edge_dim, hidden_dim)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        act_cls = get_act_fn(activation)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 act_cls(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 act_cls())
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

        self.edge_out_linear = nn.Linear(hidden_dim, edge_dim)

    def forward(self, x, z):
        ### x = self.node_input_linear(x).transpose(0, 1)
        x_in = x[:,None,:,:] + x[:,:,None,:]
        z = self.edge_input_linear(z) + x_in
        z_residual = z
        z = self.layer_norm_1(z)
        z = self.mlp(z)
        z = self.layer_norm_2(z + z_residual)
        z = self.edge_out_linear(z)
        return z


class EdgeUpdaterSAM_1(nn.Module):
    """Custom edge representation updater 1."""

    def __init__(self,
            node_dim: int,
            edge_dim: int,
            edge_downsample: int = 2,
            node_module: str = "linear",
            use_ij_ln: bool = False,
            activation: str = "relu"
        ):
        super().__init__()
        hidden_dim = edge_dim // edge_downsample
        self.node_linear_i = nn.Linear(node_dim, hidden_dim)
        self.node_linear_j = nn.Linear(node_dim, hidden_dim)
        node_modules_ij = []
        if use_ij_ln:
            node_modules_ij.append(nn.LayerNorm(hidden_dim))
        if node_module == "linear":
            node_modules_ij.extend([
                nn.Linear(hidden_dim, edge_dim)
            ])
        elif node_module == "mlp":
            node_modules_ij.extend([
                nn.Linear(hidden_dim, hidden_dim),
                get_act_fn(activation)(),
                nn.Linear(hidden_dim, edge_dim),
            ])
        else:
            raise KeyError(node_module)
        self.node_module_ij = nn.Sequential(*node_modules_ij)

    def forward(self, x, z):
        n_i = self.node_linear_i(x)
        n_j = self.node_linear_j(x)
        n_in = n_i[:,None,:,:] + n_j[:,:,None,:]
        return self.node_module_ij(n_in)


class EdgeUpdaterAlphaFold2(nn.Module):
    """New module documentation: TODO."""

    def __init__(self, in_dim, out_dim):
        """Arguments: TODO."""
        super().__init__()

    def forward(self, x):
        raise NotImplementedError()


class EdgeUpdaterWrapper_v01(nn.Module):
    """New module documentation: TODO."""

    use_ln_edge_modes = ("sam_0", "sam_1", "framediff")
    use_ln_node_modes = ("sam_0", "sam_1", "framediff", "esm")
    independent_mechanism_modes = ("esm", )

    def __init__(self,
            mode: str,
            token_dim: int,
            edge_dim: int,
            activation: str = "relu",
            linear_bias: bool = True,
            params: dict = {}
        ):
        """Arguments: TODO."""
        super().__init__()

        self.mode = mode
        if mode == "sam_0":
            # self.updater = EdgeUpdaterSAM_0(
            #     node_dim=token_dim,
            #     edge_dim=edge_dim,
            #     activation=activation,
            #     **params
            # )
            # caller = lambda p, h: self.updater(x=h, z=p)
            raise NotImplementedError()
        elif mode == "sam_1":
            self.updater = EdgeUpdaterSAM_1(
                node_dim=token_dim,
                edge_dim=edge_dim,
                activation=activation,
                **params
            )
            caller = lambda p, h: self.updater(x=h, z=p)
        elif mode == "framediff":
            self.updater = EdgeUpdaterFrameDiff(
                node_dim=token_dim,
                edge_dim=edge_dim,
                activation=activation,
                **params
            )
            caller = lambda p, h: self.updater(x=h, z=p)
        elif mode == "esm":
            self.updater = EdgeUpdaterESMFold(
                node_dim=token_dim,
                edge_dim=edge_dim,
                activation=get_act_fn(activation),
            )
            caller = lambda p, h: self.updater(x=h, z=p)
        else:
            raise KeyError(mode)
        
        self.independent_mechanism = mode in self.independent_mechanism_modes
        if mode in self.use_ln_edge_modes:
            self.ln_edge = nn.LayerNorm(edge_dim)
        else:
            self.ln_edge = nn.Identity()
        if mode in self.use_ln_node_modes:
            self.ln_token = nn.LayerNorm(token_dim)
        else:
            self.ln_token = nn.Identity()

        self._updater_caller = caller
        

    def forward(self, p, h):
        p = self.ln_edge(p)
        h = self.ln_token(h)
        if not self.independent_mechanism:
            p = p + self._updater_caller(p, h)
        else:
            p = self._updater_caller(p, h)
        return p