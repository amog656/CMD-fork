import numpy as np
import torch
import torch.nn as nn

###############################################################################
# Triangle multiplicative update.                                             #
###############################################################################

from functools import partialmethod, partial
import math
from typing import Optional, List

# import torch
# import torch.nn as nn

class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.
    """
    def __init__(self, c_z, c_hidden, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_g = Linear(self.c_z, self.c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_z, init="final")

        self.layer_norm_in = nn.LayerNorm(self.c_z)
        self.layer_norm_out = nn.LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self,
        a: torch.Tensor,
        b: torch.Tensor,
        _inplace_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        if(self._outgoing):
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b,  (2, 0, 1))

        # if(_inplace_chunk_size is not None):
        #     # To be replaced by torch vmap
        #     for i in range(0, a.shape[-3], _inplace_chunk_size):
        #         a_chunk = a[..., i: i + _inplace_chunk_size, :, :]
        #         b_chunk = b[..., i: i + _inplace_chunk_size, :, :]
        #         a[..., i: i + _inplace_chunk_size, :, :] = (
        #             torch.matmul(
        #                 a_chunk,
        #                 b_chunk,
        #             )
        #         )
        #
        #     p = a
        # else:
        #     p = torch.matmul(a, b)
        p = torch.matmul(a, b)

        return permute_final_dims(p, (1, 2, 0))


    def forward(self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: Optional[int] = 256,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        # if(inplace_safe):
        #     x = self._inference_forward(
        #         z,
        #         mask,
        #         inplace_chunk_size=_inplace_chunk_size,
        #         with_add=_add_with_inplace,
        #     )
        #     return x

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = mask
        a = a * self.sigmoid(self.linear_a_g(z))
        a = a * self.linear_a_p(z)
        b = mask
        b = b * self.sigmoid(self.linear_b_g(z))
        b = b * self.linear_b_p(z)
        x = self._combine_projections(a, b)
        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x


###############################################################################
# Triangle attention.                                                         #
###############################################################################

def permute_final_dims(tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))

class TriangleAttention(nn.Module):
    def __init__(
        self, c_in, c_hidden, no_heads, starting=True, inf=1e9
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super(TriangleAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = nn.LayerNorm(self.c_in)

        self.linear = Linear(c_in, self.no_heads, bias=False, init="normal")

        self.mha = Attention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads
        )

    def forward(self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        if(not self.starting):
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            x = self._chunk(
                x,
                biases,
                chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
            )
        else:
            x = self.mha(
                q_x=x,
                kv_x=x,
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_lma=use_lma
            )

        if(not self.starting):
            x = x.transpose(-2, -3)

        return x


###############################################################################
# primitives.                                                                 #
###############################################################################

from functools import partial
import math
from typing import Optional, Callable, List, Tuple, Sequence
import numpy as np

# import deepspeed
import torch
import torch.nn as nn
from scipy.stats import truncnorm


DEFAULT_LMA_Q_CHUNK_SIZE=1024
DEFAULT_LMA_KV_CHUNK_SIZE=4096


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.
    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:
                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0
                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
        Softmax, but without automatic casting to fp32 when the input is of
        type bfloat16
    """
    # d = t.dtype
    # if(d is torch.bfloat16 and not deepspeed.utils.is_initialized()):
    #     with torch.cuda.amp.autocast(enabled=False):
    #         s = torch.nn.functional.softmax(t, dim=dim)
    # else:
    #     s = torch.nn.functional.softmax(t, dim=dim)
    s = torch.nn.functional.softmax(t, dim=dim)

    return s


#@torch.jit.script
def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = softmax_no_cast(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, init="final"
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads, init="gating"
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self,
        o: torch.Tensor,
        q_x: torch.Tensor
    ) -> torch.Tensor:
        if(self.linear_g is not None):
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        q_chunk_size: int = DEFAULT_LMA_Q_CHUNK_SIZE,
        kv_chunk_size: int = DEFAULT_LMA_KV_CHUNK_SIZE,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_memory_efficient_kernel:
                Whether to use a custom memory-efficient attention kernel.
                This should be the default choice for most. If none of the
                "use_<...>" flags are True, a stock PyTorch implementation
                is used instead
            use_lma:
                Whether to use low-memory attention (Staats & Rabe 2021). If
                none of the "use_<...>" flags are True, a stock PyTorch
                implementation is used instead
            q_chunk_size:
                Query chunk size (for LMA)
            kv_chunk_size:
                Key/Value chunk size (for LMA)
        Returns
            [*, Q, C_q] attention update
        """
        if(biases is None):
            biases = []
        if(use_lma and (q_chunk_size is None or kv_chunk_size is None)):
            raise ValueError(
                "If use_lma is specified, q_chunk_size and kv_chunk_size must "
                "be provided"
            )
        if(use_memory_efficient_kernel and use_lma):
            raise ValueError(
                "Choose one of use_memory_efficient_kernel and use_lma"
            )

        # [*, H, Q/K, C_hidden]
        q, k, v = self._prep_qkv(q_x, kv_x)

        # [*, Q, H, C_hidden]
#         if(use_memory_efficient_kernel):
#             if(len(biases) > 2):
#                 raise ValueError(
#                     "If use_memory_efficient_kernel is True, you may only "
#                     "provide up to two bias terms"
#                 )
#             o = attention_core(q, k, v, *((biases + [None] * 2)[:2]))
#             o = o.transpose(-2, -3)
#         elif(use_lma):
#             biases = [
#                 b.expand(b.shape[:-2] + (q_x.shape[-2],) + (kv_x.shape[-2],))
#                 for b in biases
#             ]
#             o = _lma(q, k, v, biases, q_chunk_size, kv_chunk_size)
#             o = o.transpose(-2, -3)
#         else:
#             o = _attention(q, k, v, biases)
#             o = o.transpose(-2, -3)
        o = _attention(q, k, v, biases)
        o = o.transpose(-2, -3)
        o = self._wrap_up(o, q_x)

        return o


###############################################################################
# Folding block.                                                              #
###############################################################################

class FoldingBlock(nn.Module):

    def __init__(self, in_dim, in_dim_2d, nhead,
                 activation=nn.ReLU,
                 use_tmu=True,
                 use_ta=True,
                 time_embed_dim=128,
                 bead_embed_dim=32,
                 tmu_hidden_dim=64,
                 ta_hidden_dim=16,
                 ta_nhead=4,
                ):
        """
        in_dim: dimension of the tokens of input sequence s.
        in_dim_2d: dimension of the tokens of the pair
            representation z.
        use_tmu: use triangle multiplicative updates.
        use_ta: use triangle attention.
        """
        super().__init__()
        self.b_lin = nn.Linear(in_dim_2d, nhead)
        self.multihead_attention = Attention(c_q=in_dim,
                                             c_k=in_dim,
                                             c_v=in_dim,
                                             c_hidden=in_dim*nhead,
                                             no_heads=nhead,
                                             gating=False)

        self.e_t_lin = nn.Linear(time_embed_dim, in_dim)
        self.e_aa_lin = nn.Linear(bead_embed_dim, in_dim)
        self.update_mlp = nn.Sequential(nn.LayerNorm(in_dim),
                                        nn.Linear(in_dim, in_dim*4),
                                        activation(),
                                        nn.Linear(in_dim*4, in_dim),
                                       )

        self.out_update_lin = nn.Linear(in_dim*2, in_dim_2d)

        self.use_tmu = use_tmu
        if self.use_tmu:
            self.tmu_out = TriangleMultiplicativeUpdate(c_z=in_dim_2d,
                                                        c_hidden=tmu_hidden_dim,
                                                        _outgoing=True)
            self.tmu_in = TriangleMultiplicativeUpdate(c_z=in_dim_2d,
                                                        c_hidden=tmu_hidden_dim,
                                                        _outgoing=False)

        self.use_ta = use_ta
        if self.use_ta:
            self.ta_out = TriangleAttention(c_in=in_dim_2d, c_hidden=ta_hidden_dim,
                                            no_heads=ta_nhead, starting=True,
                                            inf=1e9)
            self.ta_in = TriangleAttention(c_in=in_dim_2d, c_hidden=ta_hidden_dim,
                                           no_heads=ta_nhead, starting=False,
                                           inf=1e9)

        self.update_mlp_2d = nn.Sequential(nn.LayerNorm(in_dim_2d),
                                           nn.Linear(in_dim_2d, in_dim_2d*4),
                                           activation(),
                                           nn.Linear(in_dim_2d*4, in_dim_2d))

    def forward(self, s, z, a, t):

        #------------
        # Update s. -
        #------------

        # Multihead attention.
        # s = s + self.multihead_attention(s, s, s, z)
        s = s.transpose(0, 1)
        b = self.b_lin(z).transpose(2, 3).transpose(1, 2)
        s = s + self.multihead_attention(q_x=s, kv_x=s, biases=[b])

        # Transition.
        s = s + self.e_aa_lin(a.transpose(0, 1)) + self.e_t_lin(t.transpose(0, 1))
        s = s + self.update_mlp(s)

        #------------
        # Update z. -
        #------------
        
        # Outer product and difference of s.
        out_prod = s[:,None,:,:] * s[:,:,None,:]
        out_diff = s[:,None,:,:] - s[:,:,None,:]
        s = s.transpose(1, 0)
        z = z + self.out_update_lin(torch.cat([out_prod, out_diff],
                                              axis=3))

        # Triangle multiplicative updates.
        if self.use_tmu:
            z = z + self.tmu_out(z)
            z = z + self.tmu_in(z)

        # Triangle attention updates.
        if self.use_ta:
            z = z + self.ta_out(z)
            z = z + self.ta_in(z)

        # Transition.
        z = z + self.update_mlp_2d(z)

        return s, z


class EdgeUpdaterESMFold(nn.Module):
    """Edge representation updater from ESMFold."""

    def __init__(self,
                 node_dim, edge_dim,
                 activation=nn.ReLU,
                 pair_transition_n=4,
                 use_tmu=True,
                 use_ta=True,
                 tmu_hidden_dim=64,
                 ta_hidden_dim=16,
                 ta_nhead=4):

        super().__init__()
        self.node_update_lin = nn.Linear(node_dim*2, edge_dim)

        self.use_tmu = use_tmu
        if self.use_tmu:
            self.tmu_out = TriangleMultiplicativeUpdate(c_z=edge_dim,
                                                        c_hidden=tmu_hidden_dim,
                                                        _outgoing=True)
            self.tmu_in = TriangleMultiplicativeUpdate(c_z=edge_dim,
                                                        c_hidden=tmu_hidden_dim,
                                                        _outgoing=False)

        self.use_ta = use_ta
        if self.use_ta:
            self.ta_out = TriangleAttention(c_in=edge_dim, c_hidden=ta_hidden_dim,
                                            no_heads=ta_nhead, starting=True,
                                            inf=1e9)
            self.ta_in = TriangleAttention(c_in=edge_dim, c_hidden=ta_hidden_dim,
                                           no_heads=ta_nhead, starting=False,
                                           inf=1e9)

        self.update_mlp_2d = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, edge_dim*pair_transition_n),
            activation(),
            nn.Linear(edge_dim*pair_transition_n, edge_dim))

    def forward(self, x, z):
        # Outer product and difference of s.
        # s = x.transpose(0, 1)
        s = x
        out_prod = s[:,None,:,:] * s[:,:,None,:]
        out_diff = s[:,None,:,:] - s[:,:,None,:]
        # s = s.transpose(1, 0)
        z = z + self.node_update_lin(torch.cat([out_prod, out_diff],
                                               axis=3))

        # Triangle multiplicative updates.
        if self.use_tmu:
            z = z + self.tmu_out(z)
            z = z + self.tmu_in(z)

        # Triangle attention updates.
        if self.use_ta:
            z = z + self.ta_out(z)
            z = z + self.ta_in(z)

        # Transition.
        z = z + self.update_mlp_2d(z)
        return z


# class TriangleMultiplicativeUpdate(nn.Module):

#     def __init__(self, in_dim, hidden_dim=32, mode="out"):
#         super().__init__()
#         # Line 1 in alg. 11.
#         self.layer_norm_1 = nn.LayerNorm(in_dim)
#         # Line 2 in alg. 11.
#         self.lin_2_1_a = nn.Linear(in_dim, hidden_dim)
#         self.lin_2_2_a = nn.Linear(in_dim, hidden_dim)
#         self.lin_2_1_b = nn.Linear(in_dim, hidden_dim)
#         self.lin_2_2_b = nn.Linear(in_dim, hidden_dim)
#         # Line 3 in alg. 11.
#         self.lin_3 = nn.Linear(in_dim, in_dim)
#         # Line 4 in alg. 11.
#         self.lin_4 = nn.Linear(hidden_dim, in_dim)
#         self.layer_norm_4 = nn.LayerNorm(hidden_dim)
#         if mode == "in":
#             self.aggr_dim = 1
#         elif mode == "out":
#             self.aggr_dim = 2
#         else:
#             raise KeyError(mode)
#         self.mode = mode

#     def forward(self, z):
#         z = self.layer_norm_1(z)
#         a = torch.sigmoid(self.lin_2_1_a(z)) * self.lin_2_2_a(z)
#         b = torch.sigmoid(self.lin_2_1_b(z)) * self.lin_2_2_b(z)
#         g = torch.sigmoid(self.lin_3(z))
#         # x = (a*b).sum(dim=self.aggr_dim).unsqueeze(self.aggr_dim)
#         # print("a, b:", a.shape, b.shape)
#         #------------------
#         if self.mode == "out":
#             a = permute_final_dims(a, (2, 0, 1))
#             b = permute_final_dims(b, (2, 1, 0))
#         elif self.mode == "in":
#             a = permute_final_dims(a, (2, 1, 0))
#             b = permute_final_dims(b,  (2, 0, 1))
#         else:
#             raise KeyError(self.mode)
#         # print("a*, b*:", a.shape, b.shape)
#         x = torch.matmul(a, b)
#         x = permute_final_dims(x, (1, 2, 0))
#         #------------------
#         z = g * self.lin_4(self.layer_norm_4(x))
#         return z

# class TriangleAttention(nn.Module):

#     def __init__(self, in_dim, nhead, hidden_dim=16, mode="out"):
#         super().__init__()
#         self.layer_norm_1 = nn.LayerNorm(in_dim)
#         self.nhead = nhead
#         self.hidden_dim = hidden_dim
#         self.q_lin = nn.Linear(in_dim, hidden_dim*nhead, bias=False)
#         self.k_lin = nn.Linear(in_dim, hidden_dim*nhead, bias=False)
#         self.v_lin = nn.Linear(in_dim, hidden_dim*nhead, bias=False)
#         self.b_lin = nn.Linear(in_dim, nhead, bias=False)
#         self.g_lin = nn.Linear(in_dim, hidden_dim*nhead, bias=True)
#         self.mode = mode

#     def forward(self, z):
#         z = self.layer_norm_1(z)
#         q = self.q_lin(z)
#         k = self.k_lin(z)
#         v = self.v_lin(z)
#         b = self.b_lin(z)
#         g = torch.sigmoid(self.g_lin(z))

#         if 1:
#             # Reshape to (N, L, L, H, C).
#             _len = q.shape[1]
#             _num = q.shape[0]
#             def _first_reshape(t):
#                 t = t.contiguous().view(t.shape[0], t.shape[1], t.shape[2], self.nhead, self.hidden_dim)
#                 return t
#             q = _first_reshape(q)
#             k = _first_reshape(k)
#             v = _first_reshape(v)
#             print("q, k, v:", q.shape, k.shape, v.shape)

#             # Reshape from (N, L, L, H, C) to (N, L, H, L, C) -> (N*L*H, L, C)
#             def _second_reshape(t):
#                 t = t.transpose(2, 3).contiguous().view(-1, _len, self.hidden_dim)
#                 return t
#             q = _second_reshape(q)
#             q = q / np.sqrt(self.hidden_dim)
#             k = _second_reshape(k)
#             v = _second_reshape(v)
#             print("q*, k*, v*:", q.shape, k.shape, v.shape)

#             # (N*L*H, L, C) x (N*L*H, C, L) -> (N*L*H, L, L)
#             dp_aff = torch.bmm(q, k.transpose(-2, -1))
#             print("dp_aff, b:", dp_aff.shape, b.shape)
#             # (N*L*H, L, L) -> (N, L, H, L, L).
#             dp_aff = dp_aff.view(_num*self.nhead, _len, _len, _len)
#             # (N, L, L, H) -> (N, L, H, L) -> (N, H, L, L) -> (N*H, L, L).
#             b = b.transpose(2, 3).transpose(1, 2).contiguous().view(-1, _len, _len)
#             # (N*H, 1, L, L)
#             b = b.unsqueeze(1)
#             print("dp_aff*, b*:", dp_aff.shape, b.shape)
#             dp_aff = dp_aff + b

#             attn = nn.functional.softmax(dp_aff, dim=-1)
#         else:
#             pass

#         return z

if __name__ == "__main__":
    s_dim = 32
    z_dim = 16
    nhead = 2
    t_e_dim = 128
    b_e_dim = 32

    N = 128
    L = 12

    s = torch.randn(L, N, s_dim)
    z = torch.randn(N, L, L, z_dim)
    t = torch.randn(L, N, t_e_dim)
    a = torch.randn(L, N, b_e_dim)
    block = FoldingBlock(in_dim=s_dim, in_dim_2d=z_dim, nhead=nhead,
                         time_embed_dim=t_e_dim,
                         bead_embed_dim=b_e_dim)
    o_s, o_z = block(s=s, z=z, a=a, t=t)
    print(o_s.shape, o_z.shape)