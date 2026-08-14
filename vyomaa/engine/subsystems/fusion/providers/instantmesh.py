
"""InstantMesh torch.hub adapter for sparse-view surface reconstruction."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np

from ....core.exceptions import FusionEngineError
from ....registry.registry import FUSION_REGISTRY
from ..base import BaseFusionProvider


@FUSION_REGISTRY.register_module("InstantMesh")
class InstantMeshProvider(BaseFusionProvider):
    """Run a locally cached or torch.hub InstantMesh reconstruction implementation.

    The upstream repository's hub entrypoint is configurable because releases
    expose different hubconf names.  The provider never synthesizes fallback
    geometry: a model output containing vertices and triangular faces is required.
    """

    def initialize(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise FusionEngineError("InstantMesh requires the real PyTorch runtime") from exc
        repository = str(self.config.get("torch_hub_repo", "TencentARC/InstantMesh"))
        entrypoint = str(self.config.get("torch_hub_entrypoint", "instantmesh"))
        source = str(self.config.get("torch_hub_source", "github"))
        try:
            self.model = torch.hub.load(repository, entrypoint, source=source, pretrained=True, trust_repo=True)
            self.device = str(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
            if self.device.startswith("cuda") and not torch.cuda.is_available(): self.device = "cpu"
            if hasattr(self.model, "to"): self.model = self.model.to(self.device)
            if hasattr(self.model, "eval"): self.model.eval()
        except Exception as exc:
            raise FusionEngineError(f"InstantMesh model load failed ({repository}:{entrypoint}): {exc}") from exc

    @staticmethod
    def _arrays(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Extract a triangle mesh from common InstantMesh inference return values."""
        if isinstance(result, dict):
            vertices, faces = result.get("vertices"), result.get("faces")
            colors = result.get("vertex_colors") or result.get("colors")
        else:
            vertices, faces = getattr(result, "vertices", None), getattr(result, "faces", None)
            colors = getattr(result, "vertex_colors", None)
        if vertices is None or faces is None:
            raise FusionEngineError("InstantMesh output must expose vertices and faces")
        convert = lambda value: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        return convert(vertices), convert(faces), None if colors is None else convert(colors)

    def reconstruct(self, views: Any, cameras: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Invoke the upstream model's supported reconstruct or call interface."""
        try:
            import torch
            inputs = views.to(self.device)
            camera_inputs = {name: value.to(self.device) for name, value in cameras.items()}
            with torch.inference_mode():
                if hasattr(self.model, "reconstruct"):
                    result = self.model.reconstruct(inputs, camera_inputs)
                else:
                    result = self.model(inputs, camera_inputs)
            return self._arrays(result)
        except FusionEngineError:
            raise
        except Exception as exc:
            raise FusionEngineError(f"InstantMesh inference failed: {exc}") from exc

    def cleanup(self) -> None:
        """Release InstantMesh model weights before geometry refinement."""
        if hasattr(self, "model"): del self.model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass


