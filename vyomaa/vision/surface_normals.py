"""Surface normal estimation from dense depth maps using analytical finite differences."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from ..core.contracts import Camera, DepthMap
from ..core.exceptions import VisionError
from ..core.registry import MODEL_REGISTRY, ModelSpec
from ..core.types import ArtifactType, ModelCapability
from .base import BaseVisionAdapter


SURFACE_NORMALS_SPEC = ModelSpec(
    name="SurfaceNormalEstimator",
    version="1.0.0",
    capability=ModelCapability.DENSE_CORRESPONDENCE,
    input_types=[ArtifactType.DEPTH_MAP, ArtifactType.CAMERA],
    output_types=[ArtifactType.CONFIDENCE_MAP],
    estimated_vram_bytes=0,
    description="Computes analytical surface normals from metric or relative depth arrays.",
)


@MODEL_REGISTRY.register("SurfaceNormalEstimator", spec=SURFACE_NORMALS_SPEC)
class SurfaceNormalEstimator(BaseVisionAdapter):
    """Computes accurate 3D unit surface normal maps (H, W, 3) from depth and camera intrinsics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(SURFACE_NORMALS_SPEC, config)

    def initialize(self, device: str = "cpu", precision: str = "fp32") -> None:
        self.runtime_state = "loaded_resident"

    @staticmethod
    def compute_normals_from_depth(
        depth_map: np.ndarray,
        camera: Optional[Camera] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute (H, W, 3) unit normal vectors and (H, W) confidence weights from depth gradient."""
        depth = depth_map.astype(np.float32)
        h, w = depth.shape

        fx = camera.focal_length_x if camera else 1000.0
        fy = camera.focal_length_y if camera else 1000.0

        # Gradient along X and Y
        dz_dx = np.gradient(depth, axis=1)
        dz_dy = np.gradient(depth, axis=0)

        # Scale gradient by camera intrinsics to represent true metric surface slope
        nx = -dz_dx * fx
        ny = -dz_dy * fy
        nz = depth  # forward component

        normals = np.stack([nx, ny, nz], axis=-1)
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        norm[norm == 0] = 1.0
        unit_normals = (normals / norm).astype(np.float32)

        # Confidence: lower confidence near high-frequency depth discontinuities (silhouette edges)
        laplacian = np.abs(dz_dx) + np.abs(dz_dy)
        confidence = np.exp(-laplacian / (np.mean(laplacian) + 1e-6)).astype(np.float32)

        return unit_normals, confidence

    def infer(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.compute_normals_from_depth(inputs[0], kwargs.get("camera"))
