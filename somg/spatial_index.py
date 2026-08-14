"""
VYOMAAV Base Model Engine
Module: somg.spatial_index

3D Axis-Aligned Bounding Box (AABB) spatial grid index for O(1)/O(log N) geometric queries.
"""

from typing import List, Dict, Set, Tuple
from somg.entity import SOMGEntity


class AABBSpatialIndex:
    """3D Uniform Spatial Hash Grid Index for bounding box overlap queries."""

    def __init__(self, cell_size: float = 2.0):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int, int], Set[str]] = {}
        self.entity_bounds: Dict[str, Tuple[List[float], List[float]]] = {}

    def _get_cell_coords(self, point: List[float]) -> Tuple[int, int, int]:
        return (
            int(point[0] // self.cell_size),
            int(point[1] // self.cell_size),
            int(point[2] // self.cell_size)
        )

    def insert_or_update(self, entity_id: str, bbox_min: List[float], bbox_max: List[float]):
        """Inserts or updates entity bounds in the spatial index."""
        self.remove(entity_id)
        self.entity_bounds[entity_id] = (bbox_min, bbox_max)

        min_cell = self._get_cell_coords(bbox_min)
        max_cell = self._get_cell_coords(bbox_max)

        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    cell = (x, y, z)
                    if cell not in self.grid:
                        self.grid[cell] = set()
                    self.grid[cell].add(entity_id)

    def remove(self, entity_id: str):
        """Removes entity from spatial grid."""
        if entity_id in self.entity_bounds:
            bbox_min, bbox_max = self.entity_bounds[entity_id]
            min_cell = self._get_cell_coords(bbox_min)
            max_cell = self._get_cell_coords(bbox_max)

            for x in range(min_cell[0], max_cell[0] + 1):
                for y in range(min_cell[1], max_cell[1] + 1):
                    for z in range(min_cell[2], max_cell[2] + 1):
                        cell = (x, y, z)
                        if cell in self.grid and entity_id in self.grid[cell]:
                            self.grid[cell].remove(entity_id)
                            if not self.grid[cell]:
                                del self.grid[cell]
            del self.entity_bounds[entity_id]

    def query_aabb_overlap(self, query_min: List[float], query_max: List[float]) -> Set[str]:
        """Finds all candidate entity IDs whose bounding boxes overlap the query volume."""
        min_cell = self._get_cell_coords(query_min)
        max_cell = self._get_cell_coords(query_max)

        candidates: Set[str] = set()
        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    cell = (x, y, z)
                    if cell in self.grid:
                        candidates.update(self.grid[cell])

        # Exact AABB Intersection Check
        matching_entities: Set[str] = set()
        for eid in candidates:
            b_min, b_max = self.entity_bounds[eid]
            if (b_min[0] <= query_max[0] and b_max[0] >= query_min[0] and
                b_min[1] <= query_max[1] and b_max[1] >= query_min[1] and
                b_min[2] <= query_max[2] and b_max[2] >= query_min[2]):
                matching_entities.add(eid)

        return matching_entities