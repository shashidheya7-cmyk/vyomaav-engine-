"""
VYOMAAV Base Model Engine
Test Suite: tests/test_sprint6.py

Pytest suite validating Sprint 6: 3D-RoPE rotary attention, Spatial-Temporal Vision Encoder,
Lie Algebra SE(3) exponential mapping, and end-to-end integration into VYOMAAVBaseModel.
"""

import pytest
import torch
from base_model.rope3d import RotaryEmbedding3D
from base_model.vision_encoder import VYOMAAVVisionEncoder
from base_model.camera_estimator import LieAlgebraCameraEstimator
from base_model.model import VYOMAAVBaseModel
from base_model.contracts import ModelInputBatch


def test_3d_rope_tensor_shapes():
    rope = RotaryEmbedding3D(dim=24)  # 24 / 3 = 8 per axis
    q = torch.randn(2, 4, 16, 24)     # (B, H, N, head_dim)
    k = torch.randn(2, 4, 16, 24)
    coords_3d = torch.randn(2, 16, 3) # (B, N, 3) -> (x, y, z)

    q_rot, k_rot = rope(q, k, coords_3d)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.isnan(q_rot).any()


def test_vision_encoder_rgbd_fusion():
    encoder = VYOMAAVVisionEncoder(embed_dim=64, patch_size=16, num_layers=2)
    frames = torch.randn(2, 3, 3, 64, 64)       # B=2, T=3, C=3, H=64, W=64
    depths = torch.randn(2, 3, 1, 64, 64)       # B=2, T=3, C=1, H=64, W=64

    tokens = encoder(frames, depth_maps=depths)

    # 64x64 image with 16x16 patch = 4x4 = 16 patches per frame
    assert tokens.shape == (2, 3, 16, 64)


def test_lie_algebra_camera_estimator():
    estimator = LieAlgebraCameraEstimator(embed_dim=64)
    visual_tokens = torch.randn(2, 3, 16, 64)   # (B, T, N, D)

    pred_k, pred_poses = estimator(visual_tokens)

    assert pred_k.shape == (2, 3, 3, 3)
    assert pred_poses.shape == (2, 3, 3, 4)
    # Assert positive focal length constraint
    assert (pred_k[..., 0, 0] > 0.0).all()
    assert (pred_k[..., 1, 1] > 0.0).all()


def test_end_to_end_base_model_with_sprint6_backbones():
    vision = VYOMAAVVisionEncoder(embed_dim=64, patch_size=16)
    camera = LieAlgebraCameraEstimator(embed_dim=64)

    model = VYOMAAVBaseModel(
        embed_dim=64,
        num_classes=10,
        vision_encoder=vision,
        camera_estimator=camera
    )

    dummy_frames = torch.randn(1, 2, 3, 32, 32)
    batch = ModelInputBatch(frames=dummy_frames)

    outputs = model(batch)

    assert outputs.pred_intrinsics_k.shape == (1, 2, 3, 3)
    assert outputs.pred_poses_se3.shape == (1, 2, 3, 4)
    assert outputs.world_latent.shape[0] == 1