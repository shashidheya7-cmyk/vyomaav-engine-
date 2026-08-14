"""Depth Anything V2 adapter for dense monocular depth estimation."""

from __future__ import annotations

from typing import Any, Dict, Optional

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
    input_types=[
        ArtifactType.OBSERVATION,
        ArtifactType.INPUT_MEDIA,
    ],
    output_types=[
        ArtifactType.DEPTH_MAP,
    ],
    estimated_vram_bytes=int(4.0 * (1024 ** 3)),
    supported_precisions=[
        PrecisionType.FP16,
        PrecisionType.FP32,
    ],
    description=(
        "Zero-shot foundation model for high-resolution dense depth estimation."
    ),
)


@MODEL_REGISTRY.register(
    "DepthAnythingV2",
    spec=DEPTH_ANYTHING_V2_SPEC,
)
class DepthAnythingV2Adapter(BaseVisionAdapter):
    """Depth Anything V2 adapter with explicit runtime state and dense output handoff."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(DEPTH_ANYTHING_V2_SPEC, config)

        self.model = None
        self.processor = None

        # Concrete dense inference result kept in memory for the DAG
        # handoff to point-map/backprojection stages.
        self.last_depth_array: Optional[np.ndarray] = None

    def initialize(
        self,
        device: str = "cuda",
        precision: str = "fp16",
    ) -> None:
        """Load Depth Anything V2 on the explicitly requested device."""

        device = str(device).lower().strip()
        precision = str(precision).lower().strip()

        if device not in {"cuda", "cpu"}:
            raise ModelUnavailableError(
                f"Unsupported device '{device}'. Expected 'cuda' or 'cpu'."
            )

        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailableError(
                f"DepthAnythingV2 requires torch: {exc}"
            ) from exc

        if device == "cuda" and not torch.cuda.is_available():
            raise ModelUnavailableError(
                "DepthAnythingV2 requested CUDA, but CUDA is not "
                "available in the current environment."
            )

        try:
            from transformers import (
                AutoImageProcessor,
                AutoModelForDepthEstimation,
            )
        except ImportError as exc:
            raise ModelUnavailableError(
                f"DepthAnythingV2 requires transformers: {exc}"
            ) from exc

        model_id = self.config.get(
            "model_id",
            "depth-anything/Depth-Anything-V2-Large-hf",
        )

        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_id
            )

            self.model = AutoModelForDepthEstimation.from_pretrained(
                model_id
            )

            if device == "cuda":
                self.model = self.model.to("cuda")

                if precision == "fp16":
                    self.model = self.model.half()
                elif precision == "fp32":
                    self.model = self.model.float()
                else:
                    raise ModelUnavailableError(
                        f"Unsupported precision '{precision}'. "
                        "Expected 'fp16' or 'fp32'."
                    )
            else:
                self.model = self.model.to("cpu")

                # Keep CPU inference in FP32 for reliability.
                self.model = self.model.float()

            self.model.eval()
            self.last_depth_array = None
            self.runtime_state = "loaded_resident"

        except ModelUnavailableError:
            self.model = None
            self.processor = None
            self.runtime_state = "unavailable"
            raise

        except Exception as exc:
            self.model = None
            self.processor = None
            self.last_depth_array = None
            self.runtime_state = "unavailable"

            raise ModelUnavailableError(
                f"Failed to load DepthAnythingV2 weights "
                f"from '{model_id}': {exc}"
            ) from exc

    def estimate_depth(
        self,
        image_rgb: Image.Image,
        observation: Optional[Observation] = None,
    ) -> DepthMap:
        """Run dense inference and retain the full-resolution depth array."""

        if self.model is None or self.processor is None:
            raise ModelUnavailableError(
                "DepthAnythingV2 model is uninitialized. "
                "Call initialize() before inference."
            )

        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailableError(
                f"DepthAnythingV2 requires torch for inference: {exc}"
            ) from exc

        if not isinstance(image_rgb, Image.Image):
            raise VisionError(
                "DepthAnythingV2 expects a PIL.Image.Image input."
            )

        width, height = image_rgb.size

        try:
            inputs = self.processor(
                images=image_rgb,
                return_tensors="pt",
            )

            device = next(self.model.parameters()).device

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            prediction = (
                torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            prediction = np.asarray(
                prediction,
                dtype=np.float32,
            )

            if prediction.shape != (height, width):
                raise VisionError(
                    "DepthAnythingV2 produced an unexpected depth shape: "
                    f"{prediction.shape}, expected {(height, width)}."
                )

            if not np.all(np.isfinite(prediction)):
                raise VisionError(
                    "DepthAnythingV2 produced non-finite depth values."
                )

            self.last_depth_array = prediction.copy()

            min_depth = float(np.min(prediction))
            max_depth = float(np.max(prediction))

            return DepthMap(
                name=(
                    f"Depth_"
                    f"{observation.frame_id if observation else 'image'}"
                ),
                width=width,
                height=height,
                min_depth=min_depth,
                max_depth=max_depth,
                is_metric=False,
                camera_id=(
                    observation.camera.artifact_id
                    if observation and observation.camera
                    else None
                ),
                confidence_score=0.92,
                provenance=ProvenanceRecord(
                    producer_subsystem="vision",
                    producer_model="DepthAnythingV2",
                    parent_artifact_ids=(
                        [observation.artifact_id]
                        if observation
                        else []
                    ),
                ),
            )

        except (ModelUnavailableError, VisionError):
            raise

        except Exception as exc:
            self.last_depth_array = None
            raise VisionError(
                f"DepthAnythingV2 inference failed: {exc}"
            ) from exc

    def infer(
        self,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Generic inference entry point."""

        if not inputs:
            raise VisionError(
                "DepthAnythingV2.infer() requires an image input."
            )

        return self.estimate_depth(
            inputs[0],
            kwargs.get("observation"),
        )
