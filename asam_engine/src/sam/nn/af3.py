import math
import torch
import torch.nn as nn
from typing import Callable


class Transition(nn.Module):
    """Transition module from AF3."""

    def __init__(self, in_dim: int, n: int = 2, linear_bias: bool = True):
        """Arguments: TODO."""
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.a_linear = nn.Sequential(
            nn.Linear(in_dim, in_dim*n, bias=linear_bias),
            nn.SiLU()
        )
        self.b_linear = nn.Linear(in_dim, in_dim*n, bias=linear_bias)
        self.o_linear = nn.Linear(in_dim*n, in_dim, bias=linear_bias)

    def forward(self, x):
        x = self.ln(x)
        a = self.a_linear(x)
        b = self.b_linear(x)
        x = self.o_linear(a * b)
        return x


class FourierEmbedding(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
            time_dim: int,
            train_scale: bool = False
        ):
        """Arguments: TODO."""
        super().__init__()
        # Shape (1, time_dim) so that it can be broadcasted with time.
        w = torch.randn(time_dim).unsqueeze(0)
        b = torch.randn(time_dim).unsqueeze(0)
        self.register_buffer("w", w)
        self.register_buffer("b", b)
        if train_scale:
            self.time_scaler = nn.LayerNorm(1)
        else:
            self.time_scaler = nn.Identity()
        self.train_scale = train_scale

    def forward(self, t):
        """
        t: shape (B, )
        """
        t_hat = t.unsqueeze(1)  # (B, 1)
        t_hat = self.time_scaler(t_hat)
        return torch.cos(2*math.pi*(t_hat*self.w + self.b))


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale,
                              requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)



class AdaLN(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
            token_dim: int,
            node_dim: int,
            linear_bias: bool = True
        ):
        """Arguments: TODO."""
        super().__init__()
        self.h_ln = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.s_ln = nn.LayerNorm(node_dim, elementwise_affine=True)  # bias=False
        self.s_linear_1 = nn.Linear(node_dim, token_dim)
        self.s_linear_2 = nn.Linear(node_dim, token_dim, bias=linear_bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h, s):
        h = self.h_ln(h)
        s = self.s_ln(s)
        h = self.sigmoid(self.s_linear_1(s))*h + self.s_linear_2(s)
        return h


class AttentionPairBias(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
            token_dim: int,
            edge_dim: int,
            num_heads: int,
            node_dim: int = None,
            linear_bias: bool = True,
            edge_bias: bool = False
            # n: int = 2
        ):
        """Arguments: TODO."""
        super().__init__()

        if token_dim % num_heads != 0:
            raise ValueError()
        self._num_heads = num_heads
        self._head_dim = token_dim // num_heads

        if node_dim is not None:
            self.input_project = AdaLN(
                token_dim=token_dim,
                node_dim=node_dim,
                linear_bias=linear_bias
            )
        else:
            self.input_project = nn.LayerNorm(token_dim)
        self._node_dim = node_dim
        
        self.q_project = nn.Linear(token_dim, token_dim)
        self.k_project = nn.Linear(token_dim, token_dim, bias=linear_bias)
        self.v_project = nn.Linear(token_dim, token_dim, bias=linear_bias)

        self.edge_to_bias = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, num_heads, bias=edge_bias)
        )

        self.gating = nn.Sequential(
            nn.Linear(token_dim, token_dim, bias=linear_bias),
            nn.Sigmoid()
        )

        self.scale = 1/math.sqrt(token_dim // num_heads)

        self.out_linear = nn.Linear(token_dim, token_dim, bias=linear_bias)

        if node_dim is not None:
            self.output_project = nn.Sequential(
                nn.Linear(node_dim, token_dim),
                nn.Sigmoid()
            )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        if self._node_dim is not None:
            nn.init.constant_(self.output_project[0].bias, -2.0)

    def _attn_reshape(self, x):
        b_size = x.shape[0]
        seq_l = x.shape[1]
        x = x.view(b_size, seq_l, self._num_heads, self._head_dim)
        x = x.transpose(1, 2).contiguous().view(b_size*self._num_heads, seq_l, self._head_dim)
        return x
    
    def forward(self, h, p, s=None):
        # Initialize.
        b_size = h.shape[0]
        seq_l = h.shape[1]

        # Input projections.
        if s is not None:
            if self._node_dim is None:
                raise ValueError()
            h = self.input_project(h, s)
        else:
            h = self.input_project(h)
        
        # Query, key and values.
        q = self._attn_reshape(self.q_project(h))
        k = self._attn_reshape(self.k_project(h))
        v = self._attn_reshape(self.v_project(h))

        # Pair biases from edge features.
        bias = self.edge_to_bias(p)
        bias = bias.transpose(1, 3).transpose(2, 3)
        bias = bias.contiguous().view(b_size*self._num_heads, seq_l, seq_l)
        # Gating.
        g = self._attn_reshape(self.gating(h))

        # Scaled dot product and bias addition.
        aff = self.scale*(torch.bmm(q, k.transpose(1, 2))) + bias
        # Softmax to calculate attention matrix.
        attn = nn.functional.softmax(aff, dim=-1)
        # Update with the values and gating.
        y = g * torch.bmm(attn, v)
        # Reshape.
        y = y.view(b_size, self._num_heads, seq_l, self._head_dim).transpose(1, 2)
        y = y.contiguous().view(b_size, seq_l, self._num_heads*self._head_dim)
        # Output tokens.
        h = self.out_linear(y)

        # Output projection from adnLN-Zero(?).
        if s is not None:
            h = self.output_project(s) * h

        return h


class ConditionedTransition(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
            token_dim: int,
            node_dim: int,
            linear_bias: bool = True,
            n: int = 2
        ):
        """Arguments: TODO."""
        super().__init__()

        self.ada_ln = AdaLN(
            token_dim=token_dim,
            node_dim=node_dim,
            linear_bias=linear_bias
        )
        self.h_linear_1 = nn.Linear(token_dim, token_dim*n, bias=linear_bias)
        self.h_linear_2 = nn.Linear(token_dim, token_dim*n, bias=linear_bias)
        self.swish = nn.SiLU()

        self.s_linear = nn.Linear(node_dim, token_dim)
        self.b_linear = nn.Linear(token_dim*n, token_dim, bias=linear_bias)
        self.sigmoid = nn.Sigmoid()

        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.constant_(self.s_linear.bias, -2.0)

    def forward(self, h, s):
        h = self.ada_ln(h, s)
        b = self.swish(self.h_linear_1(h)) * self.h_linear_2(h)
        h = self.sigmoid(self.s_linear(s)) * self.b_linear(b)
        return h