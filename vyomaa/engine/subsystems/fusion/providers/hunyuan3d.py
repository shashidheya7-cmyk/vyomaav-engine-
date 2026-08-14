
"""Hunyuan3D real image-to-shape provider adapter."""

import gc
from typing import Any
import numpy as np
from PIL import Image

from ....core.exceptions import FusionEngineError
from ....registry.registry import FUSION_REGISTRY
from ..base import BaseFusionProvider


@FUSION_REGISTRY.register_module("Hunyuan3D")
class Hunyuan3DProvider(BaseFusionProvider):
    """Use Tencent's Hunyuan3D shape-generation pipeline without geometry fallbacks."""

    def initialize(self) -> None:
        try:
            import torch
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        except ModuleNotFoundError as exc:
            raise FusionEngineError("Hunyuan3D requires the hy3dgen package and PyTorch") from exc
        device = str(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(str(self.config.get("model_id", "tencent/Hunyuan3D-2")))
        self.pipeline = self.pipeline.to(device)

    def reconstruct(self, views: Any, cameras: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        try:
            array = views[0].detach().cpu().permute(1, 2, 0).numpy()
            output = self.pipeline(image=Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8)))
            mesh = getattr(output, "mesh", output)
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
            return vertices, faces, None
        except Exception as exc:
            raise FusionEngineError(f"Hunyuan3D inference failed: {exc}") from exc

    def cleanup(self) -> None:
        if hasattr(self, "pipeline"): del self.pipeline
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ModuleNotFoundError: pass


