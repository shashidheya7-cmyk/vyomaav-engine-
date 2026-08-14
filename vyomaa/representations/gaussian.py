"""3D Gaussian Splatting representation contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np

from ..core.base_artifact import BaseArtifact
from ..core.exceptions import SchemaValidationError
from ..core.metadata import ArtifactMetadata
from ..core.provenance import ProvenanceRecord
from ..core.types import ArtifactType


@dataclass
class GaussianRepresentation(BaseArtifact):
    """Container for 3D Gaussian primitive parameters."""

    num_gaussians: int = 0
    sh_degree: int = 3
    positions: Optional[np.ndarray] = None  # (N, 3)
    scales: Optional[np.ndarray] = None     # (N, 3)
    rotations: Optional[np.ndarray] = None  # (N, 4) quaternions
    opacities: Optional[np.ndarray] = None  # (N, 1)
    sh_features: Optional[np.ndarray] = None  # (N, K, 3)
    storage_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.GAUSSIAN_SPLAT
        if self.positions is not None:
            self.num_gaussians = len(self.positions)
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "num_gaussians": self.num_gaussians,
            "sh_degree": self.sh_degree,
            "storage_path": self.storage_path,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GaussianRepresentation:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.GAUSSIAN_SPLAT
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        return cls(**kwargs)
