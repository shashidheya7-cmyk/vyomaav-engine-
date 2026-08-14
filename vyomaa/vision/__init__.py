"""Neural perception and visual evidence extraction subsystem."""

from .base import BaseVisionAdapter
from .depth_anything import DepthAnythingV2Adapter, DEPTH_ANYTHING_V2_SPEC
from .sam2_segmentation import SAM2SegmentationAdapter, SAM2_SPEC
from .bytetrack_tracker import ByteTrackTracker, BYTETRACK_SPEC, TrackedBox
from .surface_normals import SurfaceNormalEstimator, SURFACE_NORMALS_SPEC

__all__ = [
    "BaseVisionAdapter",
    "DepthAnythingV2Adapter",
    "DEPTH_ANYTHING_V2_SPEC",
    "SAM2SegmentationAdapter",
    "SAM2_SPEC",
    "ByteTrackTracker",
    "BYTETRACK_SPEC",
    "TrackedBox",
    "SurfaceNormalEstimator",
    "SURFACE_NORMALS_SPEC",
]
