"""Signed Distance Field (SDF) and volumetric grid contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..core.base_artifact import BaseArtifact
from ..core.metadata import ArtifactMetadata
from ..core.provenance import ProvenanceRecord
from ..core.types import ArtifactType


@dataclass
class SDFVolumeRepresentation(BaseArtifact):
    """Discrete 3D grid storing signed Euclidean distance values."""

    resolution: Tuple[int, int, int] = (128, 128, 128)
    voxel_size: float = 0.01
    origin: List[float] = field(default_factory=lambda: [-0.64, -0.64, -0.64])
    storage_path: Optional[str] = None
    truncation_distance: float = 0.05

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.SDF_VOLUME
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "resolution": list(self.resolution),
            "voxel_size": self.voxel_size,
            "origin": self.origin,
            "storage_path": self.storage_path,
            "truncation_distance": self.truncation_distance,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SDFVolumeRepresentation:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.SDF_VOLUME
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        if "resolution" in kwargs and isinstance(kwargs["resolution"], list):
            kwargs["resolution"] = tuple(kwargs["resolution"])
        return cls(**kwargs)
