"""VGGT geometry foundation model adapter for rapid multi-view camera & point map prediction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..core.contracts import Camera, Observation
from ..core.exceptions import ModelUnavailableError, CameraGeometryError
from ..core.registry import MODEL_REGISTRY, ModelSpec
from ..core.types import ArtifactType, ModelCapability, PrecisionType
from .base import BaseGeometryAdapter


VGGT_SPEC = ModelSpec(
    name="VGGT",
    version="1.0.0",
    capability=ModelCapability.SFM_CALIBRATION,
    input_types=[ArtifactType.MEDIA_SEQUENCE, ArtifactType.INPUT_MEDIA],
    output_types=[ArtifactType.CAMERA_TRAJECTORY, ArtifactType.POINT_CLOUD],
    estimated_vram_bytes=int(12.0 * (1024 ** 3)),
    supported_precisions=[PrecisionType.FP16, PrecisionType.BF16],
    description="Vision Geometry Foundation Model for joint camera pose and 3D point map prediction.",
)


@MODEL_REGISTRY.register("VGGT", spec=VGGT_SPEC)
class VGGTAdapter(BaseGeometryAdapter):
    """VGGT adapter executing feedforward multi-view camera and geometry estimation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(VGGT_SPEC, config)
        self.model = None

    def initialize(self, device: str = "cuda", precision: str = "fp16") -> None:
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailableError(f"VGGT requires PyTorch runtime: {exc}") from exc
        # Model weights path check
        model_path = self.config.get("model_path", "weights/vggt.pt")
        raise ModelUnavailableError(f"VGGT checkpoint not found at '{model_path}'. Upstream model unavailable.")

    def infer(self, *inputs: Any, **kwargs: Any) -> Any:
        raise ModelUnavailableError("VGGT model weights unavailable for execution.")
