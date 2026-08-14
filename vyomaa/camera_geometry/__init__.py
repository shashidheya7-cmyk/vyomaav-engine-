"""Camera and spatial geometry foundation package."""

from .base import BaseGeometryAdapter
from .scale_estimator import ScaleEstimator, ScaleEstimate, ScaleSource, ScaleConfidence
from .bundle_adjustment import BundleAdjuster, BundleAdjustmentResult
from .vggt_adapter import VGGTAdapter, VGGT_SPEC
from .colmap_adapter import COLMAPAdapter, COLMAP_SPEC
from .dust3r_adapter import DUSt3RAdapter, DUST3R_SPEC
from .mast3r_adapter import MASt3RAdapter, MAST3R_SPEC
from .geometry_router import GeometryRouter

__all__ = [
    "BaseGeometryAdapter",
    "ScaleEstimator",
    "ScaleEstimate",
    "ScaleSource",
    "ScaleConfidence",
    "BundleAdjuster",
    "BundleAdjustmentResult",
    "VGGTAdapter",
    "VGGT_SPEC",
    "COLMAPAdapter",
    "COLMAP_SPEC",
    "DUSt3RAdapter",
    "DUSt3R_SPEC",
    "MASt3RAdapter",
    "MAST3R_SPEC",
    "GeometryRouter",
]
