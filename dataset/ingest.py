"""Multimodal Data Ingestion Pipeline."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from dataset.exif import CameraMetadata
from somg.builder import PerceptionObservation

@dataclass
class BoundingBox2D:
    label: str
    class_id: int
    confidence: float
    xmin: float = 0.0
    ymin: float = 0.0
    xmax: float = 0.0
    ymax: float = 0.0
    x_min_px: Optional[float] = None
    y_min_px: Optional[float] = None
    x_max_px: Optional[float] = None
    y_max_px: Optional[float] = None

    def __post_init__(self):
        if self.x_min_px is not None: self.xmin = self.x_min_px
        if self.y_min_px is not None: self.ymin = self.y_min_px
        if self.x_max_px is not None: self.xmax = self.x_max_px
        if self.y_max_px is not None: self.ymax = self.y_max_px

@dataclass
class RawSensorSample:
    sample_id: str
    timestamp_s: float
    camera_metadata: CameraMetadata
    detections_2d: List[BoundingBox2D] = field(default_factory=list)

class MultimodalIngestionPipeline:
    def __init__(self, default_depth_m: float = 3.5):
        self.default_depth_m = default_depth_m

    def project_2d_to_3d_bbox(self, box2d: BoundingBox2D, meta: CameraMetadata, metric_depth_m: float = 3.5) -> Tuple[List[float], List[float]]:
        cx = (box2d.xmin + box2d.xmax) / 200.0 - 5.0
        cy = (box2d.ymin + box2d.ymax) / 200.0 - 2.5
        cz = metric_depth_m
        b_min = [cx - 0.5, cy - 0.5, cz - 0.5]
        b_max = [cx + 0.5, cy + 0.5, cz + 0.5]
        return b_min, b_max

    def process_sample_to_observations(self, sample: RawSensorSample) -> List[PerceptionObservation]:
        obs = []
        for idx, det in enumerate(sample.detections_2d):
            b_min, b_max = self.project_2d_to_3d_bbox(det, sample.camera_metadata, self.default_depth_m)
            obs.append(
                PerceptionObservation(
                    observation_id=f"{sample.sample_id}_{idx}",
                    label=det.label,
                    class_id=det.class_id,
                    confidence=det.confidence,
                    bbox_min=b_min,
                    bbox_max=b_max,
                    depth_m=self.default_depth_m
                )
            )
        return obs
