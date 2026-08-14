"""MASt3R matching and geometry verification adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from ..core.exceptions import ModelUnavailableError
from ..core.registry import MODEL_REGISTRY, ModelSpec
from ..core.types import ArtifactType, ModelCapability, PrecisionType
from .base import BaseGeometryAdapter


MAST3R_SPEC = ModelSpec(
    name="MASt3R",
    version="1.0.0",
    capability=ModelCapability.DENSE_CORRESPONDENCE,
    input_types=[ArtifactType.OBSERVATION],
    output_types=[ArtifactType.CONFIDENCE_MAP, ArtifactType.POINT_CLOUD],
    estimated_vram_bytes=int(10.0 * (1024 ** 3)),
    supported_precisions=[PrecisionType.FP16, PrecisionType.BF16],
    description="Matching And Stereo 3D Reconstruction with pixel-accurate 3D feature descriptors.",
)


@MODEL_REGISTRY.register("MASt3R", spec=MAST3R_SPEC)
class MASt3RAdapter(BaseGeometryAdapter):
    """Adapter for MASt3R dense feature matching and geometry verification."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(MAST3R_SPEC, config)

    def initialize(self, device: str = "cuda", precision: str = "fp16") -> None:
        try:
            import torch
            import mast3r
        except ImportError as exc:
            raise ModelUnavailableError(f"MASt3R requires 'mast3r' package: {exc}") from exc
        self.runtime_state = "loaded_resident"

    def infer(self, *inputs: Any, **kwargs: Any) -> Any:
        raise ModelUnavailableError("MASt3R weights unavailable in local environment.")
