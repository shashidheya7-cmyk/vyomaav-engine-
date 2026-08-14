"""
VYOMAAV Base Model Engine
Test Suite: tests/test_sdf_tune.py

Pytest suite validating Sprint 21: Positional encoding, Neural Implicit SDF network shapes,
surface distance regression, and Eikonal gradient loss constraint.
"""

import pytest
import torch
from base_model.sdf_tune import PositionalEncoding, NeuralImplicitSDF


def test_positional_encoding_dimensions():
    encoder = PositionalEncoding(num_frequencies=4, include_input=True)
    x = torch.randn(2, 10, 3)
    encoded = encoder(x)

    # input_dim = 3 + 3 * 2 * 4 = 27
    assert encoded.shape == (2, 10, 27)


def test_neural_implicit_sdf_forward():
    sdf_net = NeuralImplicitSDF(hidden_dim=128, num_layers=6, skip_connections=[3])
    coords = torch.randn(4, 16, 3, requires_grad=True)

    distances = sdf_net(coords)

    assert distances.shape == (4, 16, 1)
    assert not torch.isnan(distances).any()


def test_eikonal_loss_constraint():
    sdf_net = NeuralImplicitSDF(hidden_dim=64, num_layers=4, skip_connections=[])
    coords = torch.randn(8, 3, requires_grad=True)

    eik_loss = sdf_net.compute_eikonal_loss(coords)

    assert eik_loss.item() > 0.0
    assert not torch.isnan(eik_loss)
    eik_loss.backward()
    assert coords.grad is not None