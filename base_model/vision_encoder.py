"""
VYOMAAV Base Model Engine
Module: base_model.vision_encoder

Production Vision Transformer Encoder with RGB-D patch fusion, 3D-RoPE spatial-temporal
attention layers, and multi-frame video tokenization.
"""

from typing import Optional
import torch
import torch.nn as nn
from base_model.interfaces import IVisionEncoder
from base_model.rope3d import RotaryEmbedding3D


class SpatialTemporalAttentionBlock(nn.Module):
    """Transformer block with 3D-RoPE spatial-temporal self-attention."""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.rope3d = RotaryEmbedding3D(dim=self.head_dim)

    def forward(self, x: torch.Tensor, coords_3d: torch.Tensor) -> torch.Tensor:
        # x: (B, N_tokens, D)
        # coords_3d: (B, N_tokens, 3)
        B, N, D = x.shape
        residual = x
        x_norm = self.norm1(x)

        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, N, head_dim)

        # Apply 3D Rotary Position Embeddings
        q_rot, k_rot = self.rope3d(q, k, coords_3d)

        # Scaled Dot-Product Attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.proj(out) + residual

        # MLP Block
        out = out + self.mlp(self.norm2(out))
        return out


class VYOMAAVVisionEncoder(IVisionEncoder):
    """Spatial-Temporal Vision Encoder with RGB-D patch projection and 3D-RoPE."""

    def __init__(
        self,
        embed_dim: int = 128,
        patch_size: int = 16,
        num_layers: int = 4,
        num_heads: int = 8,
        in_channels: int = 4  # 3 RGB + 1 Depth
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Patch Projection for RGB-D frames
        self.patch_conv = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.blocks = nn.ModuleList([
            SpatialTemporalAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, frames: torch.Tensor, depth_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        # frames: (B, T, 3, H, W)
        B, T, C, H, W = frames.shape

        if depth_maps is None:
            depth_maps = torch.zeros((B, T, 1, H, W), device=frames.device, dtype=frames.dtype)

        # Fuse RGB and Depth -> (B*T, 4, H, W)
        rgbd = torch.cat([frames, depth_maps], dim=2).view(B * T, 4, H, W)
        patches = self.patch_conv(rgbd)  # (B*T, D, H_p, W_p)
        H_p, W_p = patches.shape[2], patches.shape[3]
        N_p = H_p * W_p  # Patches per frame

        tokens = patches.flatten(2).transpose(1, 2)  # (B*T, N_p, D)
        tokens = tokens.reshape(B, T * N_p, self.embed_dim)  # Combine time & space: (B, T*N_p, D)

        # Generate synthetic grid coordinates (x, y, t) for 3D-RoPE
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H_p, device=frames.device),
            torch.linspace(-1, 1, W_p, device=frames.device),
            indexing="ij"
        )
        grid_2d = torch.stack([grid_x, grid_y], dim=-1).reshape(N_p, 2)  # (N_p, 2)

        coords_list = []
        for t in range(T):
            t_coord = torch.full((N_p, 1), float(t), device=frames.device)
            coords_t = torch.cat([grid_2d, t_coord], dim=-1)  # (N_p, 3)
            coords_list.append(coords_t)

        coords_3d = torch.cat(coords_list, dim=0).unsqueeze(0).expand(B, -1, -1)  # (B, T*N_p, 3)

        # Apply Transformer Blocks
        for block in self.blocks:
            tokens = block(tokens, coords_3d)

        # Reshape to standard (B, T, N_p, D)
        return tokens.view(B, T, N_p, self.embed_dim)