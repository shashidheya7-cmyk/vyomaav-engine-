import logging
from typing import Dict, List, Any, Optional
from vyomaa.multiview.contracts import ViewSet

logger = logging.getLogger("vyomaa.multiview.view_graph")

class ViewGraphNode:
    def __init__(self, observation_id: str, index: int, timestamp: float, metadata: Dict[str, Any]):
        self.observation_id = observation_id
        self.index = index
        self.timestamp = timestamp
        self.metadata = metadata

class ViewGraphEdge:
    def __init__(self, source_id: str, target_id: str, edge_type: str, weight: float, confidence: float, attributes: Dict[str, Any]):
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type
        self.weight = weight
        self.confidence = confidence
        self.attributes = attributes

class ViewGraph:
    def __init__(self):
        self.nodes: Dict[str, ViewGraphNode] = {}
        self.edges: List[ViewGraphEdge] = []

    def add_node(self, observation_id: str, index: int, timestamp: float, metadata: Optional[Dict[str, Any]] = None):
        self.nodes[observation_id] = ViewGraphNode(observation_id, index, timestamp, metadata or {})

    def add_edge(self, source_id: str, target_id: str, edge_type: str, weight: float, confidence: float, attributes: Optional[Dict[str, Any]] = None):
        self.edges.append(ViewGraphEdge(source_id, target_id, edge_type, weight, confidence, attributes or {}))

    @classmethod
    def from_view_set(cls, view_set: ViewSet, temporal_window: int = 2) -> "ViewGraph":
        graph = cls()
        ids = view_set.observation_ids
        timestamps = view_set.timestamps if view_set.timestamps else [float(i) for i in range(len(ids))]

        for i, obs_id in enumerate(ids):
            graph.add_node(obs_id, i, timestamps[i], {"image_path": view_set.image_paths[i] if i < len(view_set.image_paths) else ""})

        n = len(ids)
        for i in range(n):
            for j in range(i + 1, min(i + 1 + temporal_window, n)):
                src, tgt = ids[i], ids[j]
                time_delta = abs(timestamps[j] - timestamps[i])
                weight = 1.0 / (1.0 + time_delta)
                graph.add_edge(src, tgt, "temporal_adjacency", weight=weight, confidence=0.95, attributes={"time_delta": time_delta})

        return graph

    def get_local_neighbors(self, observation_id: str, k: int = 3) -> List[str]:
        neighbors = []
        for edge in self.edges:
            if edge.source_id == observation_id and edge.edge_type == "temporal_adjacency":
                neighbors.append(edge.target_id)
        return neighbors[:k]

class ViewPair:
    def __init__(self, source_id: str, target_id: str, score: float = 0.0):
        self.source_id = source_id
        self.target_id = target_id
        self.score = score

class ViewQualityScore:
    def __init__(self, observation_id: str, sharpness: float = 1.0, exposure: float = 1.0):
        self.observation_id = observation_id
        self.sharpness = sharpness
        self.exposure = exposure

class CorrespondenceMap:
    def __init__(self, source_id: str, target_id: str, matches: Any = None):
        self.source_id = source_id
        self.target_id = target_id
        self.matches = matches
