"""
VYOMAAV Base Model Engine
Module: dataset.ingest

Multimodal dataset ingestion framework. Synchronizes RGB frames, depth maps,
LiDAR point clouds, and camera EXIF metadata into aligned frame packages
and generates PerceptionObservation batches for SOMG temporal fusion.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from dataset.exif import CameraMetadata
from somg.builder import PerceptionObservation


@dataclass
class BoundingBox2D:
    label: str
    class_id: int
    confidence: float
    x_min_px: float
    y_min_px: float
    x_max_px: float
    y_max_px: float


@dataclass
class RawSensorSample:
    """Synchronized multi-modal sensor snapshot for a single timestep."""
    sample_id: str
    timestamp_s: float
    camera_metadata: CameraMetadata
    detections_2d: List[BoundingBox2D] = field(default_factory=list)
    depth_map_ref: Optional[str] = None
    lidar_points_ref: Optional[str] = None


class MultimodalIngestionPipeline:
    """Converts 2D image detections and metric depth buffers into 3D PerceptionObservations."""

    def __init__(self, default_depth_m: float = 3.0):
        self.default_depth_m = default_depth_m

    def project_2d_to_3d_bbox(
        self,
        box2d: BoundingBox2D,
        camera_meta: CameraMetadata,
        metric_depth_m: Optional[float] = None
    ) -> Tuple[List[float], List[float]]:
        """Projects a 2D pixel bounding box into a 3D camera-space Axis-Aligned Bounding Box (AABB).

        X = (u - c_x) * Z / f_x
        Y = (v - c_y) * Z / f_y
        """
        depth = metric_depth_m if metric_depth_m is not None else self.default_depth_m
        k = camera_meta.compute_intrinsics_k()
        f_x, f_y, c_x, c_y = k[0], k[4], k[2], k[5]

        # Calculate 3D bounds for min and max box extents
        x_min_3d = (box2d.x_min_px - c_x) * depth / f_x
        x_max_3d = (box2d.x_max_px - c_x) * depth / f_x
        y_min_3d = (box2d.y_min_px - c_y) * depth / f_y
        y_max_3d = (box2d.y_max_px - c_y) * depth / f_y

        # Depth extent estimate (±0.5m around centroid depth)
        z_min_3d = max(0.1, depth - 0.5)
        z_max_3d = depth + 0.5

        bbox_min = [min(x_min_3d, x_max_3d), min(y_min_3d, y_max_3d), z_min_3d]
        bbox_max = [max(x_min_3d, x_max_3d), max(y_min_3d, y_max_3d), z_max_3d]

        return bbox_min, bbox_max

    def process_sample_to_observations(
        self, sample: RawSensorSample
    ) -> List[PerceptionObservation]:
        """Converts a raw sensor frame into a batch of 3D PerceptionObservations."""
        observations: List[PerceptionObservation] = []

        for idx, det in enumerate(sample.detections_2d):
            b_min, b_max = self.project_2d_to_3d_bbox(det, sample.camera_metadata)

            obs = PerceptionObservation(
                observation_id=f"{sample.sample_id}_{idx}",
                label=det.label,
                class_id=det.class_id,
                confidence=det.confidence,
                bbox_min=b_min,
                bbox_max=b_max,
                estimated_mass_kg=10.0,
                material_type="generic",
                aleatoric_noise=1.0 - det.confidence
            )
            observations.append(obs)

        return observations