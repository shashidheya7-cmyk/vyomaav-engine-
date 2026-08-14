"""Typed scene graph entities representing objects, structures, and camera rigs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

from ..core.base_artifact import BaseArtifact
from ..core.metadata import ArtifactMetadata
from ..core.provenance import ProvenanceRecord
from ..core.types import ArtifactType, ConfidenceLevel, EntityType
from .bounding import BoundingBox3D
from .transform import Transform3D


@dataclass
class ObjectEntity(BaseArtifact):
    """Semantic world entity representing an individual reconstructed object or region."""

    entity_type: EntityType = EntityType.STATIC_OBJECT
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    local_transform: Transform3D = field(default_factory=Transform3D)
    bounding_box: Optional[BoundingBox3D] = field(default_factory=BoundingBox3D)

    # Resource references
    geometry_artifact_id: Optional[str] = None
    material_artifact_id: Optional[str] = None
    observation_artifact_ids: List[str] = field(default_factory=list)
    semantic_labels: List[str] = field(default_factory=list)
    is_interactive: bool = False

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.OBJECT_ENTITY
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "entity_type": self.entity_type.value if isinstance(self.entity_type, EntityType) else str(self.entity_type),
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "local_transform": self.local_transform.to_dict(),
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "geometry_artifact_id": self.geometry_artifact_id,
            "material_artifact_id": self.material_artifact_id,
            "observation_artifact_ids": self.observation_artifact_ids,
            "semantic_labels": self.semantic_labels,
            "is_interactive": self.is_interactive,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObjectEntity:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.OBJECT_ENTITY
        if "entity_type" in kwargs and isinstance(kwargs["entity_type"], str):
            kwargs["entity_type"] = EntityType(kwargs["entity_type"])
        if "local_transform" in kwargs and isinstance(kwargs["local_transform"], dict):
            kwargs["local_transform"] = Transform3D.from_dict(kwargs["local_transform"])
        if "bounding_box" in kwargs and kwargs["bounding_box"]:
            kwargs["bounding_box"] = BoundingBox3D.from_dict(kwargs["bounding_box"])
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        return cls(**kwargs)
