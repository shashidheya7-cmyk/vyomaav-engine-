
"""OpenLRM torch.hub adapter for image-conditioned triplane reconstruction."""

import gc
from typing import Any
import numpy as np

from ....core.exceptions import FusionEngineError
from ....registry.registry import FUSION_REGISTRY
from ..base import BaseFusionProvider


@FUSION_REGISTRY.register_module("OpenLRM")
class OpenLRMProvider(BaseFusionProvider):
    """Load an OpenLRM hub implementation and require its actual mesh output."""

    def initialize(self) -> None:
        try:
            import torch
            repo = str(self.config.get("torch_hub_repo", "3DTopia/OpenLRM"))
            entry = str(self.config.get("torch_hub_entrypoint", "openlrm"))
            self.model = torch.hub.load(repo, entry, source=str(self.config.get("torch_hub_source", "github")), pretrained=True, trust_repo=True)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if hasattr(self.model, "to"): self.model = self.model.to(self.device)
            if hasattr(self.model, "eval"): self.model.eval()
        except (ModuleNotFoundError, Exception) as exc:
            raise FusionEngineError(f"OpenLRM model load failed: {exc}") from exc

    def reconstruct(self, views: Any, cameras: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        try:
            import torch
            with torch.inference_mode(): output = self.model.reconstruct(views.to(self.device), {key: value.to(self.device) for key, value in cameras.items()})
            vertices = output["vertices"] if isinstance(output, dict) else output.vertices
            faces = output["faces"] if isinstance(output, dict) else output.faces
            convert = lambda value: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            return convert(vertices), convert(faces), None
        except Exception as exc:
            raise FusionEngineError(f"OpenLRM inference failed: {exc}") from exc

    def cleanup(self) -> None:
        if hasattr(self, "model"): del self.model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ModuleNotFoundError: pass


