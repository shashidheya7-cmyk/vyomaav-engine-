"""WorldGraph representing planetary/room environments, region chunks, and spatial relations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import uuid

from ..core.base_artifact import BaseArtifact
from ..core.metadata import ArtifactMetadata
from ..core.provenance import ProvenanceRecord
from ..core.types import ArtifactType, SpatialRelation
from .scene_graph import SceneGraph


@dataclass
class SpatialEdge:
    """Semantic spatial relationship edge between two entities or regions."""

    source_id: str
    target_id: str
    relation: SpatialRelation
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation.value if isinstance(self.relation, SpatialRelation) else str(self.relation),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpatialEdge:
        kwargs = dict(data)
        if "relation" in kwargs and isinstance(kwargs["relation"], str):
            kwargs["relation"] = SpatialRelation(kwargs["relation"])
        return cls(**kwargs)


@dataclass
class WorldRegion:
    """Distinct spatial region/chunk within the reconstructed world."""

    region_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "unnamed_region"
    is_indoor: bool = True
    scene_graph_ids: List[str] = field(default_factory=list)
    bounding_min: List[float] = field(default_factory=lambda: [-10.0, -10.0, -10.0])
    bounding_max: List[float] = field(default_factory=lambda: [10.0, 10.0, 10.0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "name": self.name,
            "is_indoor": self.is_indoor,
            "scene_graph_ids": self.scene_graph_ids,
            "bounding_min": self.bounding_min,
            "bounding_max": self.bounding_max,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorldRegion:
        return cls(**data)


@dataclass
class WorldGraph(BaseArtifact):
    """Master orchestrator graph connecting regions, sub-scenes, and spatial relations."""

    regions: Dict[str, WorldRegion] = field(default_factory=dict)
    scene_graphs: Dict[str, SceneGraph] = field(default_factory=dict)
    spatial_edges: List[SpatialEdge] = field(default_factory=list)
    static_environment_ids: List[str] = field(default_factory=list)
    dynamic_entity_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.WORLD_GRAPH
        super().__post_init__()

    def add_region(self, region: WorldRegion) -> None:
        self.regions[region.region_id] = region

    def add_scene_graph(self, scene_graph: SceneGraph) -> None:
        self.scene_graphs[scene_graph.artifact_id] = scene_graph

    def add_spatial_relation(self, source_id: str, target_id: str, relation: SpatialRelation, confidence: float = 1.0) -> None:
        edge = SpatialEdge(source_id=source_id, target_id=target_id, relation=relation, confidence=confidence)
        self.spatial_edges.append(edge)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "regions": {rid: r.to_dict() for rid, r in self.regions.items()},
            "scene_graphs": {sid: sg.to_dict() for sid, sg in self.scene_graphs.items()},
            "spatial_edges": [e.to_dict() for e in self.spatial_edges],
            "static_environment_ids": self.static_environment_ids,
            "dynamic_entity_ids": self.dynamic_entity_ids,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorldGraph:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.WORLD_GRAPH
        if "regions" in kwargs:
            kwargs["regions"] = {rid: WorldRegion.from_dict(r) for rid, r in kwargs["regions"].items()}
        if "scene_graphs" in kwargs:
            kwargs["scene_graphs"] = {sid: SceneGraph.from_dict(sg) for sid, sg in kwargs["scene_graphs"].items()}
        if "spatial_edges" in kwargs:
            kwargs["spatial_edges"] = [SpatialEdge.from_dict(e) for e in kwargs["spatial_edges"]]
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        return cls(**kwargs)
