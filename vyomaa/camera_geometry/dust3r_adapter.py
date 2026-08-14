"""DUSt3R dense geometric correspondence and reconstruction adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from ..core.exceptions import ModelUnavailableError
from ..core.registry import MODEL_REGISTRY, ModelSpec
from ..core.types import ArtifactType, ModelCapability, PrecisionType
from .base import BaseGeometryAdapter


DUST3R_SPEC = ModelSpec(
    name="DUSt3R",
    version="1.0.0",
    capability=ModelCapability.DENSE_CORRESPONDENCE,
    input_types=[ArtifactType.OBSERVATION, ArtifactType.INPUT_MEDIA],
    output_types=[ArtifactType.POINT_CLOUD, ArtifactType.CAMERA],
    estimated_vram_bytes=int(8.0 * (1024 ** 3)),
    supported_precisions=[PrecisionType.FP16, PrecisionType.FP32],
    description="Dense Unconstrained Stereo 3D Reconstruction from uncalibrated image pairs.",
)


@MODEL_REGISTRY.register("DUSt3R", spec=DUST3R_SPEC)
class DUSt3RAdapter(BaseGeometryAdapter):
    """Adapter for DUSt3R pointmap and camera estimation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(DUST3R_SPEC, config)

    def initialize(self, device: str = "cuda", precision: str = "fp16") -> None:
        try:
            import torch
            import dust3r
        except ImportError as exc:
            raise ModelUnavailableError(f"DUSt3R requires 'dust3r' package: {exc}") from exc
        self.runtime_state = "loaded_resident"

    def infer(self, *inputs: Any, **kwargs: Any) -> Any:
        raise ModelUnavailableError("DUSt3R weights unavailable in local environment.")
