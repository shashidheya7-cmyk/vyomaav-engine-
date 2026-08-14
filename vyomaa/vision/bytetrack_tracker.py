"""ByteTrack multi-object tracking adapter for video and temporal frame sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..core.contracts import FrameArtifact
from ..core.exceptions import ModelUnavailableError, VisionError
from ..core.registry import MODEL_REGISTRY, ModelSpec
from ..core.types import ArtifactType, ModelCapability, PrecisionType
from .base import BaseVisionAdapter


@dataclass
class TrackedBox:
    """Bounding box detection with persistent tracking ID."""
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int = 0


BYTETRACK_SPEC = ModelSpec(
    name="ByteTrack",
    version="1.0.0",
    capability=ModelCapability.OBJECT_TRACKING,
    input_types=[ArtifactType.FRAME],
    output_types=[ArtifactType.FRAME],
    estimated_vram_bytes=int(1.0 * (1024 ** 3)),
    description="Multi-object association and tracking across consecutive video frames.",
)


@MODEL_REGISTRY.register("ByteTrack", spec=BYTETRACK_SPEC)
class ByteTrackTracker(BaseVisionAdapter):
    """Associates detections across temporal video frames to ensure entity identity permanence."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(BYTETRACK_SPEC, config)
        self._next_id = 1
        self._active_tracks: Dict[int, List[float]] = {}

    def initialize(self, device: str = "cpu", precision: str = "fp32") -> None:
        self.runtime_state = "loaded_resident"

    def track_frame(
        self,
        frame_artifact: FrameArtifact,
        detections: List[Tuple[List[float], float]],  # list of ([x1, y1, x2, y2], confidence)
    ) -> List[TrackedBox]:
        """Associate detections with existing tracks using IoU matching."""
        if not detections:
            return []

        results: List[TrackedBox] = []
        unmatched_dets = list(range(len(detections)))
        matched_tracks = set()

        # Simple greedy IoU matching for deterministic temporal linking
        for det_idx in list(unmatched_dets):
            bbox, score = detections[det_idx]
            best_iou = 0.0
            best_track_id = None

            for tid, last_bbox in self._active_tracks.items():
                if tid in matched_tracks:
                    continue
                iou = self._compute_iou(bbox, last_bbox)
                if iou > best_iou and iou >= 0.3:
                    best_iou = iou
                    best_track_id = tid

            if best_track_id is not None:
                self._active_tracks[best_track_id] = bbox
                matched_tracks.add(best_track_id)
                unmatched_dets.remove(det_idx)
                results.append(TrackedBox(track_id=best_track_id, bbox=bbox, confidence=score))

        # Assign new tracks
        for det_idx in unmatched_dets:
            bbox, score = detections[det_idx]
            new_id = self._next_id
            self._next_id += 1
            self._active_tracks[new_id] = bbox
            results.append(TrackedBox(track_id=new_id, bbox=bbox, confidence=score))

        return results

    @staticmethod
    def _compute_iou(boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

        denom = boxAArea + boxBArea - interArea
        return interArea / denom if denom > 0 else 0.0

    def infer(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.track_frame(inputs[0], kwargs.get("detections", []))
