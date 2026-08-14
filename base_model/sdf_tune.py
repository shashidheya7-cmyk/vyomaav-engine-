"""
VYOMAAV Base Model Engine
Module: base_model.sdf_tune

Neural Implicit Signed Distance Function (SDF) Fine-Tuning Engine (Sprint 21).
Implements a coordinate-based Multi-Layer Perceptron (MLP) with Fourier positional encoding
to fine-tune continuous, watertight 3D implicit surfaces from sparse SOMG object points.
"""

import math
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fourier feature positional encoding for high-frequency neural surface details."""

    def __init__(self, num_frequencies: int = 6, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Frequency bands
        freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 3)
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq * math.pi))
            out.append(torch.cos(x * freq * math.pi))
        return torch.cat(out, dim=-1)


class NeuralImplicitSDF(nn.Module):
    """Implicit Neural SDF Network mapping 3D coordinates (x,y,z) to signed distances."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_connections: List[int] = [4],
        num_frequencies: int = 6
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(num_frequencies=num_frequencies)
        
        input_dim = 3 + 3 * 2 * num_frequencies if self.pos_encoder.include_input else 3 * 2 * num_frequencies

        self.num_layers = num_layers
        self.skip_connections = skip_connections

        layers = []
        curr_dim = input_dim
        for i in range(num_layers):
            if i in skip_connections:
                curr_dim = input_dim + hidden_dim

            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            curr_dim = hidden_dim

        self.net = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Predicts signed distance d for 3D coordinates: (..., 3) -> (..., 1)."""
        x = self.pos_encoder(coords)
        input_x = x

        layer_idx = 0
        for i in range(self.num_layers):
            if i in self.skip_connections:
                x = torch.cat([x, input_x], dim=-1)

            x = self.net[layer_idx](x)
            x = self.net[layer_idx + 1](x)
            x = self.net[layer_idx + 2](x)
            layer_idx += 3

        return self.head(x)

    def compute_eikonal_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """Enforces Eikonal constraint ||grad f(x)|| = 1 across sampled space.

        Eikonal Equation:
        $$\|\nabla f(\mathbf{x})\| = 1$$
        """
        coords.requires_grad_(True)
        distances = self.forward(coords)
        grad_outputs = torch.ones_like(distances, device=coords.device)

        gradients = torch.autograd.grad(
            outputs=distances,
            inputs=coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        grad_norms = torch.norm(gradients, dim=-1)
        eikonal_loss = torch.mean((grad_norms - 1.0) ** 2)
        return eikonal_loss