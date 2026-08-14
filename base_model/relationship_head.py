"""
VYOMAAV Base Model Engine
Module: base_model.relationship_head

Predicts spatial and semantic relational edges (supported_by, contains, adjacent_to, etc.)
directly between all M x M latent entity pairs in world space.
"""

import torch
import torch.nn as nn


class NeuralRelationshipHead(nn.Module):
    """Predicts pairwise edge relationship logits matrix between latent entity slots."""

    def __init__(self, embed_dim: int = 128, num_relations: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_relations = num_relations

        # Pairwise feature extraction: [z_i, z_j, z_i - z_j, z_i * z_j] -> 4 * D
        self.pair_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_relations)
        )

    def forward(self, world_latent: torch.Tensor) -> torch.Tensor:
        """Predicts pairwise relationship tensor.

        Args:
            world_latent: (B, M, D) -> World memory latent slots

        Returns:
            rel_logits: (B, M, M, num_relations) -> Relation class logits for each (i, j) entity pair
        """
        B, M, D = world_latent.shape

        # Expand slots into all (i, j) pairs
        z_i = world_latent.unsqueeze(2).expand(B, M, M, D)  # (B, M, M, D)
        z_j = world_latent.unsqueeze(1).expand(B, M, M, D)  # (B, M, M, D)

        z_diff = z_i - z_j
        z_prod = z_i * z_j

        # Concatenate pairwise features: (B, M, M, 4*D)
        pair_features = torch.cat([z_i, z_j, z_diff, z_prod], dim=-1)

        # Predict relation logits for each pair
        rel_logits = self.pair_mlp(pair_features)  # (B, M, M, num_relations)

        return rel_logits