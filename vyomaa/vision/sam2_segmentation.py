"""SAM2 (Segment Anything Model 2) adapter for promptable and zero-shot panoptic segmentation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from ..core.contracts import Observation, SegmentationMask
from ..core.exceptions import ModelUnavailableError, VisionError
from ..core.provenance import ProvenanceRecord
from ..core.registry import MODEL_REGISTRY, ModelSpec
from ..core.types import ArtifactType, ModelCapability, PrecisionType
from .base import BaseVisionAdapter


SAM2_SPEC = ModelSpec(
    name="SAM2",
    version="2.0.0",
    capability=ModelCapability.PANOPTIC_SEGMENTATION,
    input_types=[ArtifactType.OBSERVATION, ArtifactType.FRAME],
    output_types=[ArtifactType.SEGMENTATION_MASK],
    estimated_vram_bytes=int(6.0 * (1024 ** 3)),
    supported_precisions=[PrecisionType.FP16, PrecisionType.BF16],
    description="Segment Anything Model 2 for zero-shot object, part, and video mask extraction.",
)


@MODEL_REGISTRY.register("SAM2", spec=SAM2_SPEC)
class SAM2SegmentationAdapter(BaseVisionAdapter):
    """Adapter executing Segment Anything 2 for object and panoptic segmentation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(SAM2_SPEC, config)
        self.predictor = None

    def initialize(self, device: str = "cuda", precision: str = "fp16") -> None:
        """Load SAM2 checkpoint into memory or raise ModelUnavailableError."""
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ModelUnavailableError(
                f"SAM2 requires 'sam2' and 'torch' packages. Missing dependency: {exc}"
            ) from exc

        model_cfg = self.config.get("model_cfg", "sam2_hiera_l.yaml")
        checkpoint = self.config.get("checkpoint_path", "checkpoints/sam2_hiera_large.pt")
        try:
            sam_model = build_sam2(model_cfg, checkpoint, device=device)
            self.predictor = SAM2ImagePredictor(sam_model)
            self.runtime_state = "loaded_resident"
        except Exception as exc:
            raise ModelUnavailableError(f"Failed to load SAM2 from '{checkpoint}': {exc}") from exc

    def segment_image(self, image_rgb: Image.Image, observation: Optional[Observation] = None) -> SegmentationMask:
        """Execute automated mask generation across image."""
        if self.predictor is None:
            raise ModelUnavailableError("SAM2 predictor is uninitialized. Call initialize() before inference.")

        w, h = image_rgb.size
        try:
            img_arr = np.array(image_rgb)
            self.predictor.set_image(img_arr)
            # Automatic grid prompt or center prompt
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array([[w//2, h//2]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )

            mask_artifact = SegmentationMask(
                name=f"Mask_{observation.frame_id if observation else 'image'}",
                width=w,
                height=h,
                num_instances=len(masks),
                class_labels=["foreground_entity", "part_a", "part_b"][:len(masks)],
                frame_id=observation.frame_id if observation else None,
                confidence_score=float(np.mean(scores)),
                provenance=ProvenanceRecord(
                    producer_subsystem="vision",
                    producer_model="SAM2",
                    parent_artifact_ids=[observation.artifact_id] if observation else [],
                ),
            )
            return mask_artifact
        except Exception as exc:
            raise VisionError(f"SAM2 segmentation failed: {exc}") from exc

    def infer(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.segment_image(inputs[0], kwargs.get("observation"))
