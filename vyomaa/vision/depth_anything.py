"""Depth Anything V2 adapter for dense monocular metric and relative depth estimation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image

from ..core.contracts import DepthMap, Observation
from ..core.exceptions import ModelUnavailableError, VisionError
from ..core.provenance import ProvenanceRecord
from ..core.registry import MODEL_REGISTRY, ModelSpec
from ..core.types import ArtifactType, ModelCapability, PrecisionType
from .base import BaseVisionAdapter


DEPTH_ANYTHING_V2_SPEC = ModelSpec(
    name="DepthAnythingV2",
    version="2.0.0",
    capability=ModelCapability.MONOCULAR_DEPTH,
    input_types=[ArtifactType.OBSERVATION, ArtifactType.INPUT_MEDIA],
    output_types=[ArtifactType.DEPTH_MAP],
    estimated_vram_bytes=int(4.0 * (1024 ** 3)),
    supported_precisions=[PrecisionType.FP16, PrecisionType.FP32],
    description="Zero-shot foundation model for high-resolution dense depth estimation.",
)


@MODEL_REGISTRY.register("DepthAnythingV2", spec=DEPTH_ANYTHING_V2_SPEC)
class DepthAnythingV2Adapter(BaseVisionAdapter):
    """Adapter executing Depth Anything V2 with hardware telemetry and typed error isolation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(DEPTH_ANYTHING_V2_SPEC, config)
        self.model = None

    def initialize(self, device: str = "cuda", precision: str = "fp16") -> None:
        """Load Depth Anything V2 weights into VRAM or raise ModelUnavailableError."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise ModelUnavailableError(
                f"DepthAnythingV2 requires 'torch' and 'transformers'. Missing dependency: {exc}"
            ) from exc

        model_id = self.config.get("model_id", "depth-anything/Depth-Anything-V2-Large-hf")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                if precision == "fp16":
                    self.model = self.model.half()
            self.model.eval()
            self.runtime_state = "loaded_resident"
        except Exception as exc:
            raise ModelUnavailableError(f"Failed to load DepthAnythingV2 weights from '{model_id}': {exc}") from exc

    def estimate_depth(self, image_rgb: Image.Image, observation: Optional[Observation] = None) -> DepthMap:
        """Run dense depth inference on an RGB image."""
        if self.model is None:
            raise ModelUnavailableError("DepthAnythingV2 model is uninitialized. Call initialize() before inference.")

        import torch
        w, h = image_rgb.size
        try:
            inputs = self.processor(images=image_rgb, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to original resolution
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

            min_d, max_d = float(np.min(prediction)), float(np.max(prediction))

            depth_artifact = DepthMap(
                name=f"Depth_{observation.frame_id if observation else 'image'}",
                width=w,
                height=h,
                min_depth=min_d,
                max_depth=max_d,
                is_metric=False,
                camera_id=observation.camera.artifact_id if observation and observation.camera else None,
                confidence_score=0.92,
                provenance=ProvenanceRecord(
                    producer_subsystem="vision",
                    producer_model="DepthAnythingV2",
                    parent_artifact_ids=[observation.artifact_id] if observation else [],
                ),
            )
            return depth_artifact
        except Exception as exc:
            raise VisionError(f"DepthAnythingV2 inference failed: {exc}") from exc

    def infer(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.estimate_depth(inputs[0], kwargs.get("observation"))
