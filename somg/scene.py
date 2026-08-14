"""SOMG Scene Engine."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from somg.graph import SpatialGraph
from somg.camera import CameraTrajectoryGraph

@dataclass
class DeltaLayer:
    layer_id: str = "default_delta"
    delta_id: Optional[str] = None
    added_entities: Dict[str, Any] = field(default_factory=dict)
    updated_entities: Dict[str, Any] = field(default_factory=dict)
    removed_entity_ids: Set[str] = field(default_factory=set)
    added_nodes: List[str] = field(default_factory=list)
    removed_nodes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.delta_id is not None:
            self.layer_id = self.delta_id

class SceneState:
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.base_graph = SpatialGraph()
        self.delta_layers: List[DeltaLayer] = []
        self.camera_graph = CameraTrajectoryGraph()

    def resolve_active_graph(self) -> SpatialGraph:
        graph = SpatialGraph()
        for node_id, node in self.base_graph.nodes.items():
            graph.add_node(node)
        for src, edges in self.base_graph.outgoing_edges.items():
            graph.outgoing_edges[src] = list(edges)

        for delta in self.delta_layers:
            for entity_id, entity in delta.added_entities.items():
                graph.add_node(entity)
            for entity_id, entity in delta.updated_entities.items():
                graph.add_node(entity)
            for entity_id in delta.removed_entity_ids:
                if entity_id in graph.nodes:
                    del graph.nodes[entity_id]
        return graph

    def push_delta(self, delta: DeltaLayer):
        self.delta_layers.append(delta)
