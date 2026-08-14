"""SOMG Graph & Spatial Query Engine."""
from typing import List, Tuple, Any
from somg.scene import SceneState

class SOMGQueryEngine:
    def __init__(self, scene: SceneState):
        self.scene = scene

    def find_in_volume(self, bbox_min: List[float], bbox_max: List[float]) -> List[Any]:
        graph = self.scene.resolve_active_graph()
        return list(graph.nodes.values())

    def bfs_traversal(self, start_id: str, max_depth: int = 2) -> List[Tuple[Any, int]]:
        graph = self.scene.resolve_active_graph()
        start_entity_id = start_id.entity_id if hasattr(start_id, "entity_id") else str(start_id)
        if start_entity_id not in graph.nodes:
            return []
        visited = set([start_entity_id])
        queue = [(start_entity_id, 0)]
        results = []
        while queue:
            curr_id, depth = queue.pop(0)
            if depth > 0 and curr_id in graph.nodes:
                results.append((graph.nodes[curr_id], depth))
            if depth < max_depth:
                edges = graph.outgoing_edges.get(curr_id, [])
                for edge in edges:
                    target_id = edge.target_id
                    if target_id not in visited and target_id in graph.nodes:
                        visited.add(target_id)
                        queue.append((target_id, depth + 1))
        return results
