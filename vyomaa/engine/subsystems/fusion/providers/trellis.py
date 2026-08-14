
"""Microsoft TRELLIS image-to-3D reconstruction adapter."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
from PIL import Image

from ....core.exceptions import FusionEngineError
from ....registry.registry import FUSION_REGISTRY
from ..base import BaseFusionProvider


@FUSION_REGISTRY.register_module("TRELLIS")
class TRELLISProvider(BaseFusionProvider):
    """Run Microsoft's image-conditioned TRELLIS pipeline with real weights."""

    def initialize(self) -> None:
        try:
            import torch
            from trellis.pipelines import TrellisImageTo3DPipeline
        except ModuleNotFoundError as exc:
            raise FusionEngineError("TRELLIS requires its upstream package and PyTorch; install TRELLIS from source") from exc
        self._torch = torch
        self.device = str(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.startswith("cuda") and not torch.cuda.is_available(): self.device = "cpu"
        model_id = str(self.config.get("model_id", "microsoft/TRELLIS-image-large"))
        try:
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model_id).to(self.device)
        except Exception as exc:
            raise FusionEngineError(f"TRELLIS model load failed ({model_id}): {exc}") from exc

    @staticmethod
    def _to_pil(views: Any) -> Image.Image:
        """Use the front canonical view as TRELLIS's image-conditioned input."""
        array = views[0].detach().cpu().permute(1, 2, 0).numpy()
        return Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8), "RGB")

    @staticmethod
    def _arrays(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        meshes = result.get("meshes") if isinstance(result, dict) else getattr(result, "meshes", None)
        mesh = meshes[0] if meshes else result
        vertices, faces = getattr(mesh, "vertices", None), getattr(mesh, "faces", None)
        if vertices is None or faces is None: raise FusionEngineError("TRELLIS output did not contain a triangular mesh")
        convert = lambda value: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        colors = getattr(mesh, "vertex_colors", None)
        return convert(vertices), convert(faces), None if colors is None else convert(colors)

    def reconstruct(self, views: Any, cameras: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Execute TRELLIS with configured sampling controls."""
        try:
            result = self.pipeline.run(self._to_pil(views), seed=int(self.config.get("seed", 42)), sparse_structure_sampler_params={"steps": int(self.config.get("sparse_steps", 12))}, slat_sampler_params={"steps": int(self.config.get("slat_steps", 12))})
            return self._arrays(result)
        except FusionEngineError:
            raise
        except Exception as exc:
            raise FusionEngineError(f"TRELLIS inference failed: {exc}") from exc

    def cleanup(self) -> None:
        """Release TRELLIS latent/diffusion weights after reconstruction."""
        if hasattr(self, "pipeline"): del self.pipeline
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass


