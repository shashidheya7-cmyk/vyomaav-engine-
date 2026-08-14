"""Explicit scale state estimation distinguishing metric, relative, and unknown bounds."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ScaleSource(str, Enum):
    """Origin of scale information."""
    UNKNOWN = "unknown"
    RELATIVE_NORMALIZED = "relative_normalized"
    ESTIMATED_OBJECT_PRIOR = "estimated_object_prior"
    SENSOR_METRIC_RGBD = "sensor_metric_rgbd"
    GROUND_TRUTH_METRIC = "ground_truth_metric"


class ScaleConfidence(str, Enum):
    """Reliability of scale estimate."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXACT = "exact"


@dataclass
class ScaleEstimate:
    """Explicit scale factor and uncertainty tracking container."""

    scale_factor_to_meters: float = 1.0
    source: ScaleSource = ScaleSource.RELATIVE_NORMALIZED
    confidence: ScaleConfidence = ScaleConfidence.LOW
    confidence_score: float = 0.5
    reference_object_class: Optional[str] = None
    bounding_box_diagonal_meters: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale_factor_to_meters": self.scale_factor_to_meters,
            "source": self.source.value,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "reference_object_class": self.reference_object_class,
            "bounding_box_diagonal_meters": self.bounding_box_diagonal_meters,
        }


class ScaleEstimator:
    """Infers absolute metric scale from RGB-D sensors, semantic object priors, or relative bounds."""

    @staticmethod
    def estimate_scale(
        is_metric_sensor: bool = False,
        semantic_class: Optional[str] = None,
        observed_bounds_extent: float = 1.0,
    ) -> ScaleEstimate:
        """Derive grounded scale estimate without inventing meters."""
        if is_metric_sensor:
            return ScaleEstimate(
                scale_factor_to_meters=1.0,
                source=ScaleSource.SENSOR_METRIC_RGBD,
                confidence=ScaleConfidence.EXACT,
                confidence_score=0.98,
                bounding_box_diagonal_meters=observed_bounds_extent,
            )

        # Approximate metric priors for known standard objects
        standard_sizes_meters = {
            "cup": 0.12,
            "chair": 0.85,
            "table": 1.20,
            "laptop": 0.35,
            "car": 4.50,
            "human": 1.70,
            "bottle": 0.25,
        }

        if semantic_class and semantic_class.lower() in standard_sizes_meters:
            expected_sz = standard_sizes_meters[semantic_class.lower()]
            scale_factor = expected_sz / max(observed_bounds_extent, 1e-4)
            return ScaleEstimate(
                scale_factor_to_meters=float(scale_factor),
                source=ScaleSource.ESTIMATED_OBJECT_PRIOR,
                confidence=ScaleConfidence.MEDIUM,
                confidence_score=0.75,
                reference_object_class=semantic_class,
                bounding_box_diagonal_meters=expected_sz,
            )

        # Monocular ambiguous scale
        return ScaleEstimate(
            scale_factor_to_meters=1.0,
            source=ScaleSource.RELATIVE_NORMALIZED,
            confidence=ScaleConfidence.LOW,
            confidence_score=0.35,
        )
