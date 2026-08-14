"""Hierarchical SceneGraph maintaining entity relationships, cameras, and observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from ..core.base_artifact import BaseArtifact
from ..core.contracts import Camera, Observation
from ..core.exceptions import SceneGraphError
from ..core.metadata import ArtifactMetadata
from ..core.provenance import ProvenanceRecord
from ..core.types import ArtifactType
from .entity import ObjectEntity
from .transform import Transform3D


@dataclass
class SceneGraph(BaseArtifact):
    """Hierarchical directed tree of entities, spatial transformations, and sensor observations."""

    root_id: str = "root_node"
    entities: Dict[str, ObjectEntity] = field(default_factory=dict)
    cameras: Dict[str, Camera] = field(default_factory=dict)
    observations: Dict[str, Observation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.SCENE_GRAPH
        if self.root_id not in self.entities:
            # Create a default root anchor
            root = ObjectEntity(
                artifact_id=self.root_id,
                name="Root World Anchor",
                confidence_score=1.0,
            )
            self.entities[self.root_id] = root
        super().__post_init__()

    def add_entity(self, entity: ObjectEntity, parent_id: Optional[str] = None) -> None:
        """Insert an entity node into the hierarchical graph."""
        target_parent = parent_id or self.root_id
        if target_parent not in self.entities:
            raise SceneGraphError(f"Target parent entity '{target_parent}' does not exist in SceneGraph")

        entity.parent_id = target_parent
        self.entities[entity.artifact_id] = entity
        if entity.artifact_id not in self.entities[target_parent].child_ids:
            self.entities[target_parent].child_ids.append(entity.artifact_id)

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity and re-parent its children to its parent."""
        if entity_id == self.root_id:
            raise SceneGraphError("Cannot remove the root node of the SceneGraph")
        if entity_id not in self.entities:
            raise SceneGraphError(f"Entity '{entity_id}' not found in SceneGraph")

        entity = self.entities[entity_id]
        parent = self.entities.get(entity.parent_id) if entity.parent_id else None

        if parent:
            if entity_id in parent.child_ids:
                parent.child_ids.remove(entity_id)
            # Re-parent children
            for child_id in entity.child_ids:
                if child_id in self.entities:
                    self.entities[child_id].parent_id = parent.artifact_id
                    if child_id not in parent.child_ids:
                        parent.child_ids.append(child_id)

        del self.entities[entity_id]

    def add_camera(self, camera: Camera) -> None:
        """Register a calibrated camera contract."""
        self.cameras[camera.artifact_id] = camera

    def add_observation(self, observation: Observation) -> None:
        """Register an observation binding."""
        self.observations[observation.artifact_id] = observation

    def get_global_transform(self, entity_id: str) -> Transform3D:
        """Recursively compute the composite global world transformation for an entity."""
        if entity_id not in self.entities:
            raise SceneGraphError(f"Entity '{entity_id}' not in SceneGraph")

        chain: List[Transform3D] = []
        curr: Optional[str] = entity_id
        visited = set()

        while curr is not None:
            if curr in visited:
                raise SceneGraphError(f"Circular hierarchy loop detected at entity '{curr}'")
            visited.add(curr)
            node = self.entities.get(curr)
            if not node:
                break
            chain.append(node.local_transform)
            curr = node.parent_id

        # Multiply from root downwards: root * ... * child
        global_mat = np.eye(4, dtype=np.float32)
        for t in reversed(chain):
            global_mat = global_mat @ t.to_matrix()

        return Transform3D.from_matrix(global_mat)

    def find_entities_by_label(self, label: str) -> List[ObjectEntity]:
        """Query entities containing a specific semantic label."""
        return [e for e in self.entities.values() if label in e.semantic_labels]

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "root_id": self.root_id,
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "cameras": {cid: c.to_dict() for cid, c in self.cameras.items()},
            "observations": {oid: o.to_dict() for oid, o in self.observations.items()},
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SceneGraph:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.SCENE_GRAPH
        if "entities" in kwargs:
            kwargs["entities"] = {eid: ObjectEntity.from_dict(e) for eid, e in kwargs["entities"].items()}
        if "cameras" in kwargs:
            kwargs["cameras"] = {cid: Camera.from_dict(c) for cid, c in kwargs["cameras"].items()}
        if "observations" in kwargs:
            kwargs["observations"] = {oid: Observation.from_dict(o) for oid, o in kwargs["observations"].items()}
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        return cls(**kwargs)
