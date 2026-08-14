"""Physically-Based Rendering (PBR) metallic-roughness material contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.base_artifact import BaseArtifact
from ..core.exceptions import SchemaValidationError
from ..core.metadata import ArtifactMetadata
from ..core.provenance import ProvenanceRecord
from ..core.types import ArtifactType


@dataclass
class PBRMaterial(BaseArtifact):
    """Standard glTF/USD compatible metallic-roughness PBR material."""

    base_color_factor: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    metallic_factor: float = 0.0
    roughness_factor: float = 0.5
    emissive_factor: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    normal_scale: float = 1.0
    occlusion_strength: float = 1.0
    double_sided: bool = False
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5

    # Texture artifact references
    albedo_texture_id: Optional[str] = None
    roughness_texture_id: Optional[str] = None
    metallic_texture_id: Optional[str] = None
    normal_texture_id: Optional[str] = None
    occlusion_texture_id: Optional[str] = None
    emissive_texture_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.PBR_MATERIAL
        if not (0.0 <= self.metallic_factor <= 1.0):
            raise SchemaValidationError(f"metallic_factor must be in [0, 1], got {self.metallic_factor}")
        if not (0.0 <= self.roughness_factor <= 1.0):
            raise SchemaValidationError(f"roughness_factor must be in [0, 1], got {self.roughness_factor}")
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "base_color_factor": self.base_color_factor,
            "metallic_factor": self.metallic_factor,
            "roughness_factor": self.roughness_factor,
            "emissive_factor": self.emissive_factor,
            "normal_scale": self.normal_scale,
            "occlusion_strength": self.occlusion_strength,
            "double_sided": self.double_sided,
            "alpha_mode": self.alpha_mode,
            "alpha_cutoff": self.alpha_cutoff,
            "albedo_texture_id": self.albedo_texture_id,
            "roughness_texture_id": self.roughness_texture_id,
            "metallic_texture_id": self.metallic_texture_id,
            "normal_texture_id": self.normal_texture_id,
            "occlusion_texture_id": self.occlusion_texture_id,
            "emissive_texture_id": self.emissive_texture_id,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PBRMaterial:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.PBR_MATERIAL
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        return cls(**kwargs)
