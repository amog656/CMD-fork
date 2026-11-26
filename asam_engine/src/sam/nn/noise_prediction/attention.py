from typing import Callable
import torch
import torch.nn as nn
from sam.nn.common import get_act_fn, SwiGLU


class OriginalIdpSAM_AttentionBlock(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
            token_dim: int,
            bead_dim: int,
            time_dim: int,
            tem_dim: int,
            edge_dim: int,
            num_heads: int,
            mlp_dim: int = None,
            linear_bias: bool = True,
            edge_bias: bool = True,
            activation: Callable = nn.SiLU,
            use_swiglu: bool = False,
            ada_ln_zero: bool = False
        ):
        """Arguments: TODO."""
        super().__init__()

        act_cls = get_act_fn(activation)

        # Before-attention modules.
        self.ln_1 = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            act_cls(),
            nn.Linear(time_dim, 6 * token_dim, bias=True)
        )
        if bead_dim != time_dim:
            self.bead_project = nn.Linear(bead_dim, time_dim)
        else:
            self.bead_project = nn.Identity()
        self.ada_ln_zero = ada_ln_zero
        # Attention modules.
        self.edge_to_bias = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, num_heads, bias=edge_bias)
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=linear_bias,
            batch_first=True,
            # **factory_kwargs
        )
        self._num_heads = num_heads
        # After-attention modules.
        self.ln_2 = nn.LayerNorm(token_dim, elementwise_affine=False)
        mlp_dim = mlp_dim if mlp_dim is not None else token_dim*2
        if not use_swiglu:
            self.mlp = nn.Sequential(
                nn.Linear(token_dim, mlp_dim, bias=linear_bias),
                act_cls(),
                nn.Linear(mlp_dim, token_dim, bias=linear_bias)
            )
        else:
            self.mlp = SwiGLU(
                in_dim=token_dim,
                hidden_dim=mlp_dim,
                out_dim=token_dim,
                linear_bias=linear_bias
            )
        # Inject additional information.
        self.linear_init = nn.Linear(token_dim, token_dim)
        self.ln_init = nn.LayerNorm(token_dim)
        self.linear_tem = nn.Linear(tem_dim, token_dim)
        self.ln_tem = nn.LayerNorm(token_dim)

        self.initialize_weights()

    def initialize_weights(self):
        if self.ada_ln_zero:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, h, h_init, s_a, s_t, s_tem, p):
        # Initialize.
        b_size = h.shape[0]
        seq_l = h.shape[1]

        s = self.bead_project(s_a) + s_t

        # AdaNL layer.
        adaLN_m_r = self.adaLN_modulation(s).chunk(6, dim=2)
        shift_msa = adaLN_m_r[0]
        scale_msa = adaLN_m_r[1]
        gate_msa = adaLN_m_r[2]
        shift_mlp = adaLN_m_r[3]
        scale_mlp = adaLN_m_r[4]
        gate_mlp = adaLN_m_r[5]

        # Typical Transformer with pre-LN.
        h_residual = h
        h = self.ln_1(h)
        # Modulate 1.
        h = modulate(h, shift_msa, scale_msa)
        # Attention biases from the edge features.
        bias = self.edge_to_bias(p)
        bias = bias.transpose(1, 3).transpose(2, 3)
        bias = bias.contiguous().view(b_size*self._num_heads, seq_l, seq_l)
        # Self-attenion.
        h, _ = self.self_attention(
            query=h, key=h, value=h, attn_mask=bias, need_weights=False
        )
        # Gate 1.
        h = h * gate_msa  
        # Typical residual.
        h = h + h_residual

        # Typical Transformer with pre-LN.
        h_residual = h
        h = self.ln_2(h)
        # Modulate 2.
        h = modulate(h, shift_mlp, scale_mlp)
        # MLP.
        h = self.mlp(h)
        # Gate 2.
        h = h * gate_mlp
        # Typical residual.
        h = h + h_residual

        # Inject template.
        h = self.ln_tem(h + self.linear_tem(s_tem))
        # Inject input.
        h = self.ln_init(h + self.linear_init(h_init))

        return h


class IdpSAM_AttentionBlock(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
            token_dim: int,
            node_dim: int,
            edge_dim: int,
            num_heads: int,
            mlp_dim: int = None,
            linear_bias: bool = True,
            edge_bias: bool = True,
            activation: Callable = nn.SiLU,
            use_swiglu: bool = False,
            ada_ln_zero: bool = True
        ):
        """Arguments: TODO."""
        super().__init__()

        act_cls = get_act_fn(activation)

        # Before-attention modules.
        self.ln_1 = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim, bias=linear_bias),
            act_cls(),
            nn.Linear(node_dim, 6 * token_dim, bias=True)
        )
        self.ada_ln_zero = ada_ln_zero
        # Attention modules.
        self.edge_to_bias = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, num_heads, bias=edge_bias)
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=linear_bias,
            batch_first=True,
            # **factory_kwargs
        )
        self._num_heads = num_heads
        # After-attention modules.
        self.ln_2 = nn.LayerNorm(token_dim, elementwise_affine=False)
        mlp_dim = mlp_dim if mlp_dim is not None else token_dim*2
        if not use_swiglu:
            self.mlp = nn.Sequential(
                nn.Linear(token_dim, mlp_dim, bias=linear_bias),
                act_cls(),
                nn.Linear(mlp_dim, token_dim, bias=linear_bias)
            )
        else:
            self.mlp = SwiGLU(
                in_dim=token_dim,
                hidden_dim=mlp_dim,
                out_dim=token_dim,
                linear_bias=linear_bias
            )

        self.initialize_weights()
    

    def initialize_weights(self):
        if self.ada_ln_zero:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)


    def forward(self, h, s, p):
        # Initialize.
        b_size = h.shape[0]
        seq_l = h.shape[1]

        # AdaNL layer.
        adaLN_m_r = self.adaLN_modulation(s).chunk(6, dim=2)
        shift_msa = adaLN_m_r[0]
        scale_msa = adaLN_m_r[1]
        gate_msa = adaLN_m_r[2]
        shift_mlp = adaLN_m_r[3]
        scale_mlp = adaLN_m_r[4]
        gate_mlp = adaLN_m_r[5]

        # Typical Transformer with pre-LN.
        h_residual = h
        h = self.ln_1(h)
        # Modulate 1.
        h = modulate(h, shift_msa, scale_msa)
        # Attention biases from the edge features.
        bias = self.edge_to_bias(p)
        bias = bias.transpose(1, 3).transpose(2, 3)
        bias = bias.contiguous().view(b_size*self._num_heads, seq_l, seq_l)
        # Self-attenion.
        h, _ = self.self_attention(
            query=h, key=h, value=h, attn_mask=bias, need_weights=False
        )
        # Gate 1.
        h = h * gate_msa  
        # Typical residual.
        h = h + h_residual

        # Typical Transformer with pre-LN.
        h_residual = h
        h = self.ln_2(h)
        # Modulate 2.
        h = modulate(h, shift_mlp, scale_mlp)
        # MLP.
        h = self.mlp(h)
        # Gate 2.
        h = h * gate_mlp
        # Typical residual.
        h = h + h_residual

        return h


def modulate(x, shift, scale):
    return x * (1 + scale) + shift