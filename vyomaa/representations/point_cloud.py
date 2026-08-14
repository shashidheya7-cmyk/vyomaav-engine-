"""Canonical dense/sparse 3D point cloud representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np

from ..core.base_artifact import BaseArtifact
from ..core.exceptions import SchemaValidationError
from ..core.metadata import ArtifactMetadata
from ..core.provenance import ProvenanceRecord
from ..core.types import ArtifactType


@dataclass
class PointCloud(BaseArtifact):
    """Multi-attribute point cloud structure with per-point confidence."""

    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    confidence_values: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.POINT_CLOUD
        self.set_points(self.points, self.normals, self.colors, self.confidence_values)
        super().__post_init__()

    def set_points(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        confidence_values: Optional[np.ndarray] = None,
    ) -> None:
        """Validate and set contiguous point cloud arrays."""
        p = np.ascontiguousarray(points, dtype=np.float32)
        if p.ndim != 2 or p.shape[1:] != (3,):
            raise SchemaValidationError(f"points must have shape (N, 3), got {p.shape}")

        n, c, conf = None, None, None
        if normals is not None:
            n = np.ascontiguousarray(normals, dtype=np.float32)
            if n.shape != p.shape:
                raise SchemaValidationError(f"normals shape {n.shape} must match points {p.shape}")

        if colors is not None:
            c = np.ascontiguousarray(colors, dtype=np.float32)
            if c.shape != p.shape:
                raise SchemaValidationError(f"colors shape {c.shape} must match points {p.shape}")

        if confidence_values is not None:
            conf = np.ascontiguousarray(confidence_values, dtype=np.float32)
            if conf.shape != (len(p),):
                raise SchemaValidationError(f"confidence_values shape {conf.shape} must match (N,) where N={len(p)}")

        self.points, self.normals, self.colors, self.confidence_values = p, n, c, conf

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    def compute_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.points) == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return self.points.min(axis=0), self.points.max(axis=0)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "point_count": self.point_count,
            "has_normals": self.normals is not None,
            "has_colors": self.colors is not None,
            "has_confidence": self.confidence_values is not None,
            "bounds_min": self.compute_bounds()[0].tolist(),
            "bounds_max": self.compute_bounds()[1].tolist(),
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PointCloud:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.POINT_CLOUD
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        for k in ["point_count", "has_normals", "has_colors", "has_confidence", "bounds_min", "bounds_max"]:
            kwargs.pop(k, None)
        return cls(**kwargs)
