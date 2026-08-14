"""3D Rotary Position Embeddings (3D-RoPE)."""
import torch
import torch.nn as nn
from typing import Optional, Any, Tuple

class RotaryEmbedding3D(nn.Module):
    def __init__(self, dim: int = 24, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base

    def forward(self, q: torch.Tensor, k: Optional[torch.Tensor] = None, spatial_coords: Optional[torch.Tensor] = None) -> Any:
        if k is not None:
            return q, k
        return q
