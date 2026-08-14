"""Depth Anything V2 adapter for dense monocular metric and relative depth estimation."""

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
        "Zero-shot foundation model for high-resolution "
        "dense depth estimation."
    ),
)


@MODEL_REGISTRY.register(
    "DepthAnythingV2",
    spec=DEPTH_ANYTHING_V2_SPEC,
)
class DepthAnythingV2Adapter(BaseVisionAdapter):
    """Adapter executing Depth Anything V2 with strict device validation."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(DEPTH_ANYTHING_V2_SPEC, config)
        self.model = None
        self.processor = None

    def initialize(
        self,
        device: str = "cuda",
        precision: str = "fp16",
    ) -> None:
        """
        Load Depth Anything V2.

        Device contract:
        - cuda -> CUDA must actually be available.
        - cpu  -> explicitly load on CPU.
        - anything else -> ModelUnavailableError.

        Importantly, CUDA availability is checked BEFORE attempting
        Hugging Face model downloads. This keeps unavailable-GPU tests
        deterministic and prevents unnecessary weight downloads.
        """

        device = str(device).lower().strip()
        precision = str(precision).lower().strip()

        if device not in {"cuda", "cpu"}:
            raise ModelUnavailableError(
                f"Unsupported device '{device}'. "
                "Expected 'cuda' or 'cpu'."
            )

        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailableError(
                "DepthAnythingV2 requires 'torch'. "
                f"Missing dependency: {exc}"
            ) from exc

        # CRITICAL:
        # If CUDA was explicitly requested, never silently fall back
        # to CPU. The caller asked for a GPU-resident model.
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
                "DepthAnythingV2 requires 'transformers'. "
                f"Missing dependency: {exc}"
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
                # Explicit CPU mode.
                self.model = self.model.to("cpu")

                if precision == "fp32":
                    self.model = self.model.float()
                elif precision == "fp16":
                    # CPU FP16 inference is generally undesirable.
                    # Keep CPU execution in FP32 for reliability.
                    self.model = self.model.float()
                else:
                    raise ModelUnavailableError(
                        f"Unsupported precision '{precision}'. "
                        "Expected 'fp16' or 'fp32'."
                    )

            self.model.eval()
            self.runtime_state = "loaded_resident"

        except ModelUnavailableError:
            raise

        except Exception as exc:
            self.model = None
            self.processor = None
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
        """Run dense depth inference on an RGB image."""

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

        w, h = image_rgb.size

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

            # Interpolate prediction back to original resolution.
            prediction = (
                torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )

            prediction = np.asarray(
                prediction,
                dtype=np.float32,
            )

            if prediction.ndim != 2:
                raise VisionError(
                    "DepthAnythingV2 produced an invalid depth tensor "
                    f"with shape {prediction.shape}."
                )

            if not np.all(np.isfinite(prediction)):
                raise VisionError(
                    "DepthAnythingV2 produced non-finite depth values."
                )

            min_d = float(np.min(prediction))
            max_d = float(np.max(prediction))

            depth_artifact = DepthMap(
                name=(
                    f"Depth_"
                    f"{observation.frame_id if observation else 'image'}"
                ),
                width=w,
                height=h,
                min_depth=min_d,
                max_depth=max_d,
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

            return depth_artifact

        except (ModelUnavailableError, VisionError):
            raise

        except Exception as exc:
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
