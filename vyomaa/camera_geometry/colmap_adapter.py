"""COLMAP Structure-from-Motion (SfM) adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import subprocess
import numpy as np

from ..core.contracts import Camera, CameraTrajectory, Observation
from ..core.exceptions import CameraGeometryError, ModelUnavailableError
from ..core.registry import MODEL_REGISTRY, ModelSpec
from ..core.types import ArtifactType, ModelCapability
from .base import BaseGeometryAdapter


COLMAP_SPEC = ModelSpec(
    name="COLMAP",
    version="3.9.0",
    capability=ModelCapability.SFM_CALIBRATION,
    input_types=[ArtifactType.INPUT_MEDIA, ArtifactType.MEDIA_SEQUENCE],
    output_types=[ArtifactType.CAMERA_TRAJECTORY, ArtifactType.POINT_CLOUD],
    estimated_vram_bytes=int(4.0 * (1024 ** 3)),
    description="Classical Structure from Motion pipeline for sparse camera calibration.",
)


@MODEL_REGISTRY.register("COLMAP", spec=COLMAP_SPEC)
class COLMAPAdapter(BaseGeometryAdapter):
    """Wrapper interfacing with COLMAP executable binaries for feature extraction and sparse SfM."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(COLMAP_SPEC, config)
        self.colmap_bin = self.config.get("colmap_bin", "colmap")

    def initialize(self, device: str = "cpu", precision: str = "fp32") -> None:
        # Verify if colmap binary exists on host
        try:
            res = subprocess.run([self.colmap_bin, "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0:
                self.runtime_state = "loaded_resident"
                return
        except Exception:
            pass
        raise ModelUnavailableError("COLMAP binary executable not found on system PATH. Install COLMAP.")

    def infer(self, *inputs: Any, **kwargs: Any) -> Any:
        raise ModelUnavailableError("COLMAP binary executable is not available on this host.")
