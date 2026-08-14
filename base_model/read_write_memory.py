"""
VYOMAAV Base Model Engine
Module: base_model.read_write_memory

Object-Centric Persistent Read/Write Memory System.
Uses learned update gates G_m in [0, 1] so that unobserved or occluded memory slots
retain their persistent latent state, while re-observed slots update dynamically.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from base_model.interfaces import IWorldMemory


class ReadWriteWorldMemory(IWorldMemory):
    """Persistent Gated Read/Write World Memory Engine."""

    def __init__(self, embed_dim: int = 128, memory_slots: int = 32, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_slots = memory_slots
        self.num_heads = num_heads

        # Persistent memory state initialization vector
        self.init_latent = nn.Parameter(torch.randn(1, memory_slots, embed_dim) * 0.02)

        # Cross-Attention over visual tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        # Write Gate Head: Computes slot-wise update gate G in [0, 1]
        self.gate_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        self.pose_proj = nn.Linear(12, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        camera_poses: torch.Tensor,
        prev_memory_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Gated memory write pass.

        Args:
            visual_tokens: (B, T, N, D) -> Multi-frame visual patch tokens
            camera_poses: (B, T, 3, 4) -> Camera trajectory matrices
            prev_memory_state: Optional (B, M, D) -> Memory state from previous timestep
        """
        B, T, N, D = visual_tokens.shape

        # Initialize memory state if first timestep
        if prev_memory_state is None:
            prev_memory_state = self.init_latent.expand(B, -1, -1)

        # 1. Condition visual tokens with camera pose embeddings
        pose_flat = camera_poses.reshape(B, T, 12)
        pose_emb = self.pose_proj(pose_flat).unsqueeze(2)
        conditioned_tokens = visual_tokens + pose_emb

        # 2. Prepare Keys & Values: (B, T*N, D)
        kv = conditioned_tokens.view(B, T * N, D)
        kv_norm = self.norm_kv(kv)

        # 3. Query using previous persistent memory state: (B, M, D)
        q_norm = self.norm_query(prev_memory_state)

        # 4. Cross-Attention Delta Calculation
        delta_attn, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm)

        # 5. Compute Update Gates G_m in [0, 1] per memory slot
        update_gates = self.gate_head(delta_attn)  # (B, M, 1)

        # 6. Gated State Update: Z_new = (1 - G) * Z_old + G * (Z_old + Delta)
        updated_latent = (1.0 - update_gates) * prev_memory_state + update_gates * (prev_memory_state + delta_attn)

        # 7. Residual MLP Pass
        updated_latent = self.norm_out(updated_latent)
        updated_latent = updated_latent + self.mlp(updated_latent)

        return updated_latent  # (B, M, D)