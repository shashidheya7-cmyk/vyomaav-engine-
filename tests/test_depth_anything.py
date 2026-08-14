"""
VYOMAAV Perception Engine
Test Suite: tests/test_depth_anything.py

Standalone integration test validating Depth Anything V2 inference on real GPU hardware.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from perception.depth_anything import DepthAnythingV2Estimator


def test_depth_anything_v2_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    estimator = DepthAnythingV2Estimator(model_size="small", device=device)

    # Synthetic RGB test image (256x256)
    rgb_np = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Run depth inference
    depth_tensor = estimator(rgb_np)

    assert isinstance(depth_tensor, torch.Tensor)
    assert depth_tensor.shape == (1, 256, 256)
    assert depth_tensor.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert depth_tensor.min().item() >= 0.0
    assert depth_tensor.max().item() <= 1.0