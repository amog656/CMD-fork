import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from typing import Dict, Tuple, Optional

from equivariant_diffusion.dynamics import EGNNDynamics
from constants import dataset_params, FLOAT_TYPE, INT_TYPE


class FlowMatching(nn.Module):
    """
    Conditional Flow Matching for pharmacophore coordinate generation.
    Wraps EGNNDynamics (unchanged) to predict velocity field for coordinates
    and class logits for pharmacophore types.
    """

    def __init__(
        self,
        dynamics: EGNNDynamics,
        dataset_name: str,
        phar_nf: int,
        residue_nf: int,
        n_dims: int = 3,
        lambda_h: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        oracle_class_training: bool = False,
    ):
        super().__init__()
        self.dynamics = dynamics
        self.dataset_name = dataset_name
        self.phar_nf = phar_nf
        self.residue_nf = residue_nf
        self.n_dims = n_dims
        self.lambda_h = lambda_h
        self.oracle_class_training = oracle_class_training

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        # Normalization constants from dataset config
        ds_info = dataset_params[dataset_name]
        self.norm_x = ds_info.get('norm_values', [1.0, 4.0])[0]
        self.norm_h = ds_info.get('norm_values', [1.0, 4.0])[1]
        self.norm_bias_h = ds_info.get('norm_biases', [None, 0.0])[1]

        #  distribution of nodes - load from preprocessed data
        import os
        import numpy as np
        from equivariant_diffusion.en_diffusion import DistributionNodes
        size_dist_path = os.path.join(os.path.dirname(__file__), 'data_raw', 'processed_crossdock_noH_ca_only_temp', 'size_distribution.npy')
        if os.path.exists(size_dist_path):
            size_histogram = np.load(size_dist_path)
            self.size_distribution = DistributionNodes(size_histogram)
        else:
            # Fallback: create a dummy distribution
            from equivariant_diffusion.en_diffusion import DistributionNodes
            self.size_distribution = DistributionNodes(np.zeros((10, 10)))

        # Pharmacophore type decoder for final output
        self.phar_decoder = ds_info['phar_decoder']  # list of 8 class names
        self.phar_encoder = ds_info['phar_encoder']  # dict mapping name -> index

    def forward(
        self,
        phar: Dict[str, torch.Tensor],
        pocket: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Training forward pass.

        Args:
            phar: dict with keys:
                'x'         (N, 3)     raw pharmacophore coordinates
                'one_hot'   (N, 8)     ground-truth pharmacophore one-hot
                'mask'      (N,)       batch index per node
                'size'      (B,)       number of pharmacophore nodes per sample
            pocket: dict with keys:
                'x'         (M, 3)     pocket coordinates
                'one_hot'   (M, 20)    pocket amino-acid one-hot
                'mask'      (M,)       batch index per node
                'size'      (B,)       number of pocket nodes per sample

        Returns:
            loss: scalar tensor
            info: dict with loss components for logging
        """
        device = phar['x'].device
        B = phar['size'].shape[0]

        # --- 1. Normalize coordinates (same as diffusion pipeline) ---
        x1 = phar['x'] / self.norm_x                    # (N, 3)
        pocket_x = pocket['x'] / self.norm_x            # (M, 3)

        # Pharmacophore one-hot: keep as-is for CE target (class indices 0-7)
        h_gt = phar['one_hot']                          # (N, 8)
        h_gt_classes = h_gt.argmax(dim=-1)              # (N,) long

        # Pocket one-hot: keep real AA types (not normalized)
        pocket_h = pocket['one_hot']                    # (M, 20)

        # --- 2. Sample time per example (B, 1) ---
        t = torch.rand(B, 1, device=device)             # (B, 1)

        # --- 3. Sample noise and construct FM interpolation ---
        x0 = torch.randn_like(x1)                       # (N, 3)
        t_nodes = t[phar['mask']]                       # (N, 1) - broadcast per node
        x_t = (1 - t_nodes) * x0 + t_nodes * x1         # (N, 3)

        # --- 4. h-conditioning ---
        if self.oracle_class_training and self.training:
            # Oracle class conditioning during training: use ground-truth class labels
            h_cond = h_gt                 # (N, 8) - ground truth class one-hot
        else:
            # Standard training/inference: zeros for pharmacophore nodes
            h_cond = torch.zeros_like(h_gt)  # (N, 8)
        # Pocket nodes: real AA one-hot
        pocket_h_cond = pocket_h                        # (M, 20)

        # --- 5. Build EGNNDynamics input ---
        xh_phar_t = torch.cat([x_t, h_cond], dim=-1)    # (N, 11)
        xh_pocket = torch.cat([pocket_x, pocket_h_cond], dim=-1)  # (M, 23)

        # --- 6. EGNNDynamics forward (unchanged) ---
        # Returns: (vel_phar|h_logits), (vel_pocket|h_pocket_logits)
        # Each is concatenated [vel, h] along dim=-1
        out_phar, out_pocket = self.dynamics(
            xh_phar_t, xh_pocket, t, phar['mask'], pocket['mask']
        )
        # out_phar: (N, 11) = [vel_phar (3), h_logits (8)]
        # out_pocket: (M, 23) = [vel_pocket (3), h_pocket_logits (20)]
        vel_phar = out_phar[:, :self.n_dims]       # (N, 3)
        h_logits = out_phar[:, self.n_dims:]       # (N, 8)
        # vel_phar: (N, 3)  -- coordinate velocity
        # h_logits: (N, 8)  -- phar_decoder output, used as class logits

        # --- 7. Losses ---
        # Coordinate FM loss: MSE on velocity
        v_target = x1 - x0                              # (N, 3)
        loss_coord = F.mse_loss(vel_phar, v_target)

        # Pharmacophore type classification loss
        if self.class_weights is not None:
            loss_h = F.cross_entropy(h_logits, h_gt_classes, weight=self.class_weights)
        else:
            loss_h = F.cross_entropy(h_logits, h_gt_classes)

        # Total loss
        loss = loss_coord + self.lambda_h * loss_h

        info = {
            'loss': loss.detach(),
            'loss_coord': loss_coord.detach(),
            'loss_h': loss_h.detach(),
        }
        return loss, info

    @torch.no_grad()
    def sample_given_pocket(
        self,
        pocket: Dict[str, torch.Tensor],
        num_nodes_phar: torch.Tensor,
        num_steps: int = 50,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate pharmacophores given a static pocket using Euler integration.
        Uses zero class conditioning (standard generation).

        Args:
            pocket: dict with normalized keys 'x', 'one_hot', 'mask', 'size'
                      (pocket coordinates already in normalized space if coming
                      from the generation pipeline; we re-normalize to be safe)
            num_nodes_phar: (B,) number of pharmacophore points per sample
            num_steps: number of Euler integration steps
            device: torch device

        Returns:
            x_phar:        (N, 3) unnormalized coordinates
            h_phar:        (N, 8) one-hot pharmacophore types
            phar_mask:     (N,) batch indices
            pocket_mask:   (M,) batch indices
        """
        return self.sample_given_pocket_with_class(pocket, num_nodes_phar, num_steps, device, class_cond=None)

    @torch.no_grad()
    def sample_given_pocket_with_class(
        self,
        pocket: Dict[str, torch.Tensor],
        num_nodes_phar: torch.Tensor,
        num_steps: int = 50,
        device: Optional[torch.device] = None,
        class_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate pharmacophores given a static pocket using Euler integration
        with explicit class conditioning.

        Args:
            pocket: dict with normalized keys 'x', 'one_hot', 'mask', 'size'
                      (pocket coordinates already in normalized space if coming
                      from the generation pipeline; we re-normalize to be safe)
            num_nodes_phar: (B,) number of pharmacophore points per sample
            num_steps: number of Euler integration steps
            device: torch device
            class_cond: optional (N, 8) one-hot class conditioning for pharmacophore nodes.
                       If None, uses zero class conditioning (standard generation).
                       If provided, uses the provided class one-hot vectors.

        Returns:
            x_phar:        (N, 3) unnormalized coordinates
            h_phar:        (N, 8) one-hot pharmacophore types
            phar_mask:     (N,) batch indices
            pocket_mask:   (M,) batch indices
        """
        if device is None:
            device = pocket['x'].device

        B = pocket['size'].shape[0]

        # --- Normalize pocket (defensive; pipeline may already do this) ---
        pocket_x = pocket['x'] / self.norm_x              # (M, 3)
        pocket_h = pocket['one_hot']                      # (M, 20)

        # --- Build pharmacophore mask ---
        phar_mask = self._num_nodes_to_batch_mask(
            B, num_nodes_phar, device
        )                                                 # (N,)
        N = phar_mask.shape[0]

        # --- Initialize x at pocket COM + Gaussian noise ---
        pocket_com = scatter_mean(pocket_x, pocket['mask'], dim=0)  # (B, 3)
        x_t = pocket_com[phar_mask] + torch.randn(N, 3, device=device, dtype=FLOAT_TYPE)

        # --- h-conditioning: use provided class conditioning or zeros ---
        if class_cond is not None:
            h_cond = class_cond.to(device=device, dtype=FLOAT_TYPE)
        else:
            h_cond = torch.zeros(N, self.phar_nf, device=device, dtype=FLOAT_TYPE)

        # --- Euler integration ---
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = i * dt
            t_batch = torch.full((B, 1), t, device=device, dtype=FLOAT_TYPE)  # (B, 1)
            t_nodes = t_batch[phar_mask]  # (N, 1)

            xh_phar_t = torch.cat([x_t, h_cond], dim=-1)    # (N, 11)
            xh_pocket = torch.cat([pocket_x, pocket_h], dim=-1)  # (M, 23)

            out_phar, out_pocket = self.dynamics(
                xh_phar_t, xh_pocket, t_batch, phar_mask, pocket['mask']
            )
            # out_phar: (N, 11) = [vel_phar (3), h_logits (8)]
            vel_phar = out_phar[:, :self.n_dims]       # (N, 3)

            x_t = x_t + dt * vel_phar

        # --- Final step at t=1: predict pharmacophore type logits ---
        t_batch = torch.ones(B, 1, device=device, dtype=FLOAT_TYPE)  # (B, 1)
        t_nodes = t_batch[phar_mask]  # (N, 1)
        xh_phar_1 = torch.cat([x_t, h_cond], dim=-1)
        out_phar, out_pocket = self.dynamics(
            xh_phar_1, xh_pocket, t_batch, phar_mask, pocket['mask']
        )
        h_logits = out_phar[:, self.n_dims:]       # (N, 8)
        # h_logits: (N, 8)

        # Argmax -> one-hot
        h_pred = h_logits.argmax(dim=-1)                    # (N,) long
        h_onehot = F.one_hot(h_pred, self.phar_nf).to(FLOAT_TYPE)  # (N, 8)

        # --- Unnormalize coordinates ---
        x_final = x_t * self.norm_x                         # (N, 3)

        # Note: Pocket is fixed (not updated by EGNNDynamics in conditional mode).
        # Therefore pocket_com_after == pocket_com_original, and explicit frame
        # restoration (x_final += pocket_com_original - pocket_com_after) is a no-op.
        # Coordinates are returned in the same frame as the fixed pocket.

        return x_final, h_onehot, phar_mask, pocket['mask']

    # -------------------------------------------------------------------------
    # Helper methods (copied/adapted from EnVariationalDiffusion for standalone use)
    # -------------------------------------------------------------------------

    @staticmethod
    def _num_nodes_to_batch_mask(
        n_samples: int,
        num_nodes: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert per-sample node counts to a batch mask tensor.

        Args:
            n_samples: B
            num_nodes: (B,) or int
            device:

        Returns:
            mask: (total_N,) with values in [0, B-1]
        """
        if isinstance(num_nodes, int):
            num_nodes = torch.full((n_samples,), num_nodes, device=device, dtype=INT_TYPE)
        elif isinstance(num_nodes, torch.Tensor):
            num_nodes = num_nodes.to(device)

        sample_inds = torch.arange(n_samples, device=device, dtype=INT_TYPE)
        return torch.repeat_interleave(sample_inds, num_nodes)

    @staticmethod
    def _remove_mean_batch(x: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        """
        Remove center of mass per sample.
        """
        mean = scatter_mean(x, batch_mask, dim=0)  # (B, 3)
        return x - mean[batch_mask]