"""
VYOMAAV Base Model Engine
Module: base_model.camera_estimator

Lie Algebra SE(3) Camera Pose & Intrinsics Regressor.
Maps visual token sequences to 3x3 Intrinsics K and 3x4 Extrinsic pose matrices.
"""

from typing import Tuple
import torch
import torch.nn as nn
from base_model.interfaces import ICameraEstimator


class LieAlgebraCameraEstimator(ICameraEstimator):
    """Predicts intrinsic calibration and Lie Algebra se(3) tangent vectors -> SE(3) matrices."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # Predicts 6-DOF se(3) Lie Algebra vector: [v_x, v_y, v_z, w_x, w_y, w_z]
        self.se3_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 6)
        )

        # Predicts camera focal length offsets and principal points: [f_x, f_y, c_x, c_y]
        self.k_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4)
        )

    @staticmethod
    def _exp_map_se3(se3_vec: torch.Tensor) -> torch.Tensor:
        """Exponential mapping from 6D se(3) Lie algebra vector to 3x4 SE(3) transformation matrix."""
        # se3_vec: (B, T, 6)
        B, T, _ = se3_vec.shape
        t = se3_vec[..., :3]  # Translation vector (B, T, 3)
        w = se3_vec[..., 3:]  # Rotation vector (B, T, 3)

        # Small angle approximation for rotation matrix R
        theta = torch.norm(w, dim=-1, keepdim=True) + 1e-8
        w_hat = w / theta

        # Identity 3x3
        I = torch.eye(3, device=se3_vec.device).expand(B, T, 3, 3)

        # Skew-symmetric matrix [w]_x
        wx = torch.zeros(B, T, 3, 3, device=se3_vec.device)
        wx[..., 0, 1] = -w_hat[..., 2]
        wx[..., 0, 2] = w_hat[..., 1]
        wx[..., 1, 0] = w_hat[..., 2]
        wx[..., 1, 2] = -w_hat[..., 0]
        wx[..., 2, 0] = -w_hat[..., 1]
        wx[..., 2, 1] = w_hat[..., 0]

        # Rodrigues' formula for R
        R = I + torch.sin(theta).unsqueeze(-1) * wx + (1.0 - torch.cos(theta)).unsqueeze(-1) * torch.matmul(wx, wx)

        # Concatenate R (3x3) and t (3x1) -> (3x4)
        return torch.cat([R, t.unsqueeze(-1)], dim=-1)

    def forward(self, visual_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # visual_tokens: (B, T, N, D)
        B, T, N, D = visual_tokens.shape
        pooled = visual_tokens.mean(dim=2)  # Pool spatial tokens per frame -> (B, T, D)

        # Predict 6D se(3) tangent vectors
        se3_tangents = self.se3_head(pooled)  # (B, T, 6)
        pred_poses = self._exp_map_se3(se3_tangents)  # (B, T, 3, 4)

        # Predict Intrinsics K
        k_params = self.k_head(pooled)  # (B, T, 4) -> [f_x, f_y, c_x, c_y]
        f_x, f_y = k_params[..., 0], k_params[..., 1]
        c_x, c_y = k_params[..., 2], k_params[..., 3]

        pred_k = torch.zeros(B, T, 3, 3, device=visual_tokens.device)
        pred_k[..., 0, 0] = torch.abs(f_x) + 100.0  # Positive focal length constraint
        pred_k[..., 1, 1] = torch.abs(f_y) + 100.0
        pred_k[..., 0, 2] = c_x
        pred_k[..., 1, 2] = c_y
        pred_k[..., 2, 2] = 1.0

        return pred_k, pred_poses