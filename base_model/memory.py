"""
VYOMAAV Base Model Engine
Module: base_model.memory

Attention-Based World Memory System (IWorldMemory implementation).
Uses Multi-Head Cross-Attention to allow M persistent spatial latent slots to query
and aggregate visual tokens across temporal frame sequences.
"""

from typing import Optional
import torch
import torch.nn as nn
from base_model.interfaces import IWorldMemory


class AttentionWorldMemory(IWorldMemory):
    """Multi-Head Cross-Attention World Memory Engine."""

    def __init__(self, embed_dim: int = 128, memory_slots: int = 32, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_slots = memory_slots
        self.num_heads = num_heads

        # Persistent latent memory queries: (1, M, D)
        self.latent_slots = nn.Parameter(torch.randn(1, memory_slots, embed_dim) * 0.02)

        # Cross-Attention over visual tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # Pose conditioning projection: 3x4 pose matrix (12) -> embed_dim
        self.pose_proj = nn.Linear(12, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, visual_tokens: torch.Tensor, camera_poses: torch.Tensor) -> torch.Tensor:
        """Accumulates visual tokens into persistent memory slots using cross-attention.

        Args:
            visual_tokens: (B, T, N, D) -> Multi-frame visual patch tokens
            camera_poses: (B, T, 3, 4) -> Predicted camera trajectory matrices
        """
        B, T, N, D = visual_tokens.shape

        # 1. Condition visual tokens with camera pose embeddings
        pose_flat = camera_poses.reshape(B, T, 12)  # (B, T, 12)
        pose_emb = self.pose_proj(pose_flat).unsqueeze(2)  # (B, T, 1, D)
        conditioned_tokens = visual_tokens + pose_emb  # Broadcast addition

        # 2. Flatten visual sequence: (B, T*N, D)
        kv = conditioned_tokens.view(B, T * N, D)
        kv_norm = self.norm_kv(kv)

        # 3. Expand latent query slots across batch: (B, M, D)
        queries = self.latent_slots.expand(B, -1, -1)
        q_norm = self.norm_query(queries)

        # 4. Cross-Attention: Queries (M slots) key/value-attend to KV (T*N visual tokens)
        attn_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm)
        world_latent = queries + attn_out

        # 5. MLP Residual Block
        world_latent = self.norm_out(world_latent)
        world_latent = world_latent + self.mlp(world_latent)

        return world_latent  # (B, M, D)