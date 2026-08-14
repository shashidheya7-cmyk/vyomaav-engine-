"""
VYOMAAV Perception Engine
Module: perception.depth_anything

Depth Anything V2 Monocular Depth Estimation Wrapper.
Infers high-density relative and metric depth maps from RGB image frames
and formats tensors for SOMG 3D bounding box back-projection.
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class DepthAnythingV2Estimator(nn.Module):
    """Wrapper for Depth Anything V2 neural depth estimation models."""

    def __init__(
        self,
        model_size: str = "small",  # "small", "base", "large"
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_size = model_size

        # Hugging Face Model ID Mapping
        self.model_ids = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf"
        }
        self.model_id = self.model_ids.get(model_size, self.model_ids["small"])
        
        self.processor = None
        self.model = None
        self._is_loaded = False

    def load_weights(self):
        """Loads model weights and image processor from Hugging Face Hub."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        print(f"[VYOMAAV Perception] Loading Depth Anything V2 ({self.model_size}) onto {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self._is_loaded = True
        print("[VYOMAAV Perception] Depth Anything V2 loaded successfully.")

    def forward(self, rgb_image: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        """Infers dense relative depth map from an RGB image.

        Args:
            rgb_image: Input RGB image (H, W, 3) or PIL Image or Tensor

        Returns:
            depth_map: Torch tensor (1, H, W) containing depth values normalized [0, 1]
        """
        if not self._is_loaded:
            self.load_weights()

        if isinstance(rgb_image, torch.Tensor):
            # Convert (C, H, W) or (B, C, H, W) tensor to PIL Image
            if rgb_image.ndim == 4:
                rgb_image = rgb_image.squeeze(0)
            img_np = rgb_image.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            rgb_image = Image.fromarray(img_np)

        elif isinstance(rgb_image, np.ndarray):
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            rgb_image = Image.fromarray(rgb_image)

        # Preprocess and infer depth
        inputs = self.processor(images=rgb_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth  # (1, H_feat, W_feat)

        # Interpolate depth map back to original image dimensions
        orig_size = (rgb_image.height, rgb_image.width)
        depth_map = nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=orig_size,
            mode="bicubic",
            align_corners=False
        ).squeeze(1)  # (1, H, W)

        # Normalize depth to [0.0, 1.0] range
        d_min = depth_map.min()
        d_max = depth_map.max()
        depth_normalized = (depth_map - d_min) / (d_max - d_min + 1e-6)

        return depth_normalized