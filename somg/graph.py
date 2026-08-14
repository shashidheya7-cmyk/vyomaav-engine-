from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

class RelationType(Enum):
    SUPPORTED_BY = "supported_by"
    CONTAINS = "contains"
    ADJACENT_TO = "adjacent_to"
    BLOCKS_PATH = "blocks_path"
    ATTACHED_TO = "attached_to"

@dataclass(frozen=True)
class SpatialEdge:
    source_id: str
    target_id: str
    relation_type: RelationType

class SpatialGraph:
    def __init__(self):
        self.nodes: Dict[str, Any] = {}
        self.outgoing_edges: Dict[str, List[SpatialEdge]] = {}

    def add_node(self, entity):
        self.nodes[entity.entity_id] = entity
        if entity.entity_id not in self.outgoing_edges:
            self.outgoing_edges[entity.entity_id] = []

    def add_edge(self, source_id: str, target_id: str, relation_type: RelationType):
        if source_id in self.nodes and target_id in self.nodes:
            edge = SpatialEdge(source_id, target_id, relation_type)
            if edge not in self.outgoing_edges[source_id]:
                self.outgoing_edges[source_id].append(edge)

    def get_node(self, entity_id: str) -> Optional[Any]:
        return self.nodes.get(entity_id)
