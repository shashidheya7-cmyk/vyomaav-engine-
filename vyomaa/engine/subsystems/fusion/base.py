
"""Abstract interface and state integration for fusion providers."""

from abc import ABC, abstractmethod
import gc
from typing import Any, Optional

import numpy as np
from ...core.torch_compat import torch

from ...core.exceptions import FusionEngineError
from ...scene.mesh import MeshData
from ...scene.scene import Scene


class BaseFusionProvider(ABC):
    """Provider contract for reconstructing a mesh from canonical views."""

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Allocate or configure provider-specific reconstruction resources."""

    @abstractmethod
    def reconstruct(self, views: torch.Tensor, cameras: dict[str, torch.Tensor]
                    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return vertices, triangular faces, and optional vertex RGB colors."""

    def cleanup(self) -> None:
        """Release provider resources and all collectible CUDA cache memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _normalize(vertices: np.ndarray) -> np.ndarray:
        """Center and uniformly scale vertices into the [-1, 1]^3 cube."""
        if len(vertices) == 0:
            raise FusionEngineError("reconstruction returned no vertices")
        minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
        centered = vertices - (minimum + maximum) / 2.0
        extent = float(np.abs(centered).max())
        return centered.astype(np.float32) if extent <= np.finfo(np.float32).eps else (centered / extent).astype(np.float32)

    def __call__(self, scene: Scene) -> Scene:
        """Extract scene view state, reconstruct, normalize, and store its mesh."""
        if not scene.views:
            raise FusionEngineError("fusion requires at least one scene view")
        try:
            images = [view.image_tensor.detach().to(dtype=torch.float32, device="cpu") for view in scene.views]
            shape = images[0].shape
            if any(image.shape != shape for image in images):
                raise FusionEngineError("all fusion views must have matching CxHxW shape")
            views = torch.stack(images, dim=0)
            cameras = {
                "K": torch.from_numpy(np.stack([view.camera.intrinsic_matrix for view in scene.views])),
                "RT": torch.from_numpy(np.stack([view.camera.extrinsic_matrix for view in scene.views])),
            }
            if not self._initialized:
                self.initialize()
                self._initialized = True
            vertices, faces, colors = self.reconstruct(views, cameras)
            mesh = MeshData(self._normalize(np.asarray(vertices)), np.asarray(faces), colors)
            scene.geometry.set_raw_mesh(mesh)
            return scene
        except FusionEngineError:
            raise
        except Exception as exc:
            raise FusionEngineError(f"{type(self).__name__} reconstruction failed: {exc}") from exc
        finally:
            self.cleanup()
            self._initialized = False


