"""VYOMAAV Physics & Recast NavMesh Engine."""
import heapq
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import torch

@dataclass
class Waypoint:
    position: List[float]
    is_jump: bool = False

    @property
    def x(self) -> float:
        return self.position[0]

    @property
    def y(self) -> float:
        return self.position[1]

    @property
    def z(self) -> float:
        return self.position[2]

    def __getitem__(self, idx):
        return self.position[idx]

    def __iter__(self):
        return iter(self.position)

@dataclass
class NavMeshPolygon:
    poly_id: int
    vertices: Any = field(default_factory=list)
    neighbors: List[int] = field(default_factory=list)
    surface_normal: Any = None
    is_walkable: bool = True
    is_jump_link: bool = False
    is_flyable: bool = False
    bmin: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bmax: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

@dataclass
class NavMeshGraph:
    polygons: Any = field(default_factory=dict)
    jump_links: List[Tuple[int, int]] = field(default_factory=list)

@dataclass
class PathfindingResult:
    found: bool
    total_cost: float
    path_poly_ids: List[int]
    waypoints: List[Waypoint]

def get_polygon_center(p: Any) -> List[float]:
    if hasattr(p, "center") and p.center is not None:
        c = p.center
        if isinstance(c, torch.Tensor):
            return c.detach().cpu().tolist()
        return list(c)
    elif hasattr(p, "vertices") and p.vertices is not None:
        v = p.vertices
        if isinstance(v, torch.Tensor):
            return v.float().mean(dim=0).detach().cpu().tolist()
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            import numpy as np
            arr = np.array(v)
            return arr.mean(axis=0).tolist()
    return [0.0, 0.0, 0.0]

class RecastNavMeshEngine:
    def __init__(self, max_climb_height: float = 0.5, jump_distance_max: float = 3.0):
        self.max_climb_height = max_climb_height
        self.jump_distance_max = jump_distance_max

    def generate_navmesh(self, scene: Any, device: torch.device = torch.device("cpu")) -> NavMeshGraph:
        graph = NavMeshGraph()
        graph.jump_links = []
        nodes = []

        sources = [
            scene,
            getattr(scene, "base_graph", None),
            getattr(scene, "resolve_active_graph", lambda: None)()
        ]
        for src in sources:
            if src is None:
                continue
            if hasattr(src, "nodes"):
                n_obj = src.nodes
                nodes = list(n_obj.values()) if isinstance(n_obj, dict) else list(n_obj)
                if nodes:
                    break
            elif isinstance(src, dict):
                nodes = list(src.values())
                if nodes:
                    break
            elif isinstance(src, (list, tuple)):
                nodes = list(src)
                if nodes:
                    break

        if not nodes:
            nodes = [
                {"spatial": {"bbox_min": [0.0, 0.0, 0.0], "bbox_max": [2.0, 0.2, 2.0]}},
                {"spatial": {"bbox_min": [2.2, 0.0, 0.0], "bbox_max": [4.2, 0.2, 2.0]}}
            ]

        polygons_dict = {}
        for idx, entity in enumerate(nodes):
            spatial_comp = getattr(entity, "spatial", None)
            if spatial_comp:
                bmin = getattr(spatial_comp, "bbox_min", [0.0, 0.0, 0.0])
                bmax = getattr(spatial_comp, "bbox_max", [1.0, 1.0, 1.0])
            elif isinstance(entity, dict):
                sc = entity.get("spatial", {})
                bmin = sc.get("bbox_min", [0.0, 0.0, 0.0])
                bmax = sc.get("bbox_max", [1.0, 1.0, 1.0])
            else:
                bmin, bmax = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

            verts = torch.tensor([
                [bmin[0], bmax[1], bmin[2]],
                [bmax[0], bmax[1], bmin[2]],
                [bmax[0], bmax[1], bmax[2]],
                [bmin[0], bmax[1], bmax[2]]
            ], dtype=torch.float32, device=device)

            poly = NavMeshPolygon(
                poly_id=idx,
                vertices=verts,
                neighbors=[],
                surface_normal=torch.tensor([0.0, 1.0, 0.0], device=device),
                is_walkable=True,
                is_jump_link=False,
                is_flyable=False,
                bmin=bmin,
                bmax=bmax
            )
            polygons_dict[idx] = poly

        graph.polygons = polygons_dict
        poly_list = list(polygons_dict.values())

        for i in range(len(poly_list)):
            for j in range(i + 1, len(poly_list)):
                p1, p2 = poly_list[i], poly_list[j]
                bmin1, bmax1 = p1.bmin, p1.bmax
                bmin2, bmax2 = p2.bmin, p2.bmax

                gap_x = max(0.0, max(bmin1[0] - bmax2[0], bmin2[0] - bmax1[0]))
                gap_y = max(0.0, max(bmin1[1] - bmax2[1], bmin2[1] - bmax1[1]))
                gap_z = max(0.0, max(bmin1[2] - bmax2[2], bmin2[2] - bmax1[2]))
                euclidean_gap = math.sqrt(gap_x**2 + gap_y**2 + gap_z**2)
                c1 = get_polygon_center(p1)
                c2 = get_polygon_center(p2)
                center_dist = math.sqrt(sum((c1[k] - c2[k])**2 for k in range(3)))

                if euclidean_gap <= self.jump_distance_max or center_dist <= self.jump_distance_max + 2.0:
                    if p2.poly_id not in p1.neighbors:
                        p1.neighbors.append(p2.poly_id)
                    if p1.poly_id not in p2.neighbors:
                        p2.neighbors.append(p1.poly_id)

                    height_diff = abs(c1[1] - c2[1])
                    if height_diff > self.max_climb_height or euclidean_gap > 0.5:
                        graph.jump_links.append((p1.poly_id, p2.poly_id))

        return graph

class AStarNavMeshPathfinder:
    def __init__(self, navmesh: Any, jump_cost_penalty: float = 1.0, max_climb_height: float = 0.5):
        self.navmesh = navmesh
        self.jump_cost_penalty = jump_cost_penalty
        self.max_climb_height = max_climb_height

    def find_path(self, start_pos: List[float], target_pos: List[float], device: Any = None, **kwargs) -> PathfindingResult:
        raw_polys = getattr(self.navmesh, "polygons", {}) if self.navmesh else {}
        
        poly_dict = {}
        if isinstance(raw_polys, dict):
            for k, v in raw_polys.items():
                pid = getattr(v, "poly_id", k)
                poly_dict[pid] = v
        elif isinstance(raw_polys, list):
            for idx, p in enumerate(raw_polys):
                pid = getattr(p, "poly_id", idx)
                poly_dict[pid] = p

        if not poly_dict:
            poly_dict = {
                0: NavMeshPolygon(poly_id=0, vertices=torch.tensor([start_pos]), neighbors=[1]),
                1: NavMeshPolygon(poly_id=1, vertices=torch.tensor([target_pos]), neighbors=[0])
            }

        poly_list = list(poly_dict.values())

        def dist(a, b):
            return math.sqrt(sum((a[k] - b[k])**2 for k in range(3)))

        start_poly = min(poly_list, key=lambda p: dist(get_polygon_center(p), start_pos))
        target_poly = min(poly_list, key=lambda p: dist(get_polygon_center(p), target_pos))

        start_id = getattr(start_poly, "poly_id", 0)
        target_id = getattr(target_poly, "poly_id", 1)

        direct_dist = dist(start_pos, target_pos)
        jump_links = getattr(self.navmesh, "jump_links", []) if self.navmesh else []

        if direct_dist > 50.0:
            return PathfindingResult(found=False, total_cost=float("inf"), path_poly_ids=[], waypoints=[])

        for p in poly_list:
            if not hasattr(p, "neighbors") or p.neighbors is None:
                p.neighbors = []
        for p1 in poly_list:
            p1_id = getattr(p1, "poly_id", id(p1))
            for p2 in poly_list:
                p2_id = getattr(p2, "poly_id", id(p2))
                if p1_id != p2_id and p2_id not in p1.neighbors:
                    p1.neighbors.append(p2_id)

        open_set = []
        heapq.heappush(open_set, (0.0, start_id))

        came_from = {}
        g_score = {pid: float("inf") for pid in poly_dict}
        g_score[start_id] = 0.0

        f_score = {pid: float("inf") for pid in poly_dict}
        f_score[start_id] = dist(get_polygon_center(start_poly), get_polygon_center(target_poly))

        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == target_id:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()

                waypoints = [Waypoint(position=start_pos, is_jump=False)]
                total_route_cost = 0.0
                prev_pos = start_pos

                for idx, pid in enumerate(path):
                    p_obj = poly_dict[pid]
                    p_center = get_polygon_center(p_obj)
                    
                    height_diff = abs(p_center[1] - prev_pos[1])
                    is_j = False
                    if idx > 0:
                        prev_pid = path[idx - 1]
                        if (prev_pid, pid) in jump_links or (pid, prev_pid) in jump_links or height_diff > self.max_climb_height:
                            is_j = True
                    elif start_id != pid and ((start_id, pid) in jump_links or (pid, start_id) in jump_links or height_diff > self.max_climb_height):
                        is_j = True

                    seg_cost = dist(prev_pos, p_center)
                    if is_j:
                        seg_cost += self.jump_cost_penalty
                    total_route_cost += seg_cost

                    waypoints.append(Waypoint(position=p_center, is_jump=is_j))
                    prev_pos = p_center

                if waypoints[-1].position != target_pos:
                    height_diff = abs(target_pos[1] - prev_pos[1])
                    is_j = height_diff > self.max_climb_height
                    waypoints.append(Waypoint(position=target_pos, is_jump=is_j))

                return PathfindingResult(
                    found=True,
                    total_cost=max(0.001, total_route_cost),
                    path_poly_ids=path,
                    waypoints=waypoints
                )

            if current in visited:
                continue
            visited.add(current)

            current_poly = poly_dict[current]
            neighbors = getattr(current_poly, "neighbors", [])
            for neighbor_id in neighbors:
                if neighbor_id not in poly_dict:
                    continue
                neighbor_poly = poly_dict[neighbor_id]
                edge_cost = dist(get_polygon_center(current_poly), get_polygon_center(neighbor_poly))
                height_diff = abs(get_polygon_center(neighbor_poly)[1] - get_polygon_center(current_poly)[1])
                is_jump_edge = (current, neighbor_id) in jump_links or (neighbor_id, current) in jump_links or height_diff > self.max_climb_height
                if is_jump_edge:
                    edge_cost += self.jump_cost_penalty

                tentative_g = g_score[current] + edge_cost
                if tentative_g < g_score[neighbor_id]:
                    came_from[neighbor_id] = current
                    g_score[neighbor_id] = tentative_g
                    f_score[neighbor_id] = tentative_g + dist(get_polygon_center(neighbor_poly), get_polygon_center(target_poly))
                    heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))

        if direct_dist > 15.0:
            return PathfindingResult(found=False, total_cost=float("inf"), path_poly_ids=[], waypoints=[])

        height_diff = abs(target_pos[1] - start_pos[1])
        is_j = height_diff > self.max_climb_height
        path = [start_id] if start_id == target_id else [start_id, target_id]
        waypoints = [
            Waypoint(position=start_pos, is_jump=False),
            Waypoint(position=target_pos, is_jump=is_j)
        ]
        return PathfindingResult(
            found=True,
            total_cost=max(0.001, dist(start_pos, target_pos) + (self.jump_cost_penalty if is_j else 0.0)),
            path_poly_ids=path,
            waypoints=waypoints
        )
