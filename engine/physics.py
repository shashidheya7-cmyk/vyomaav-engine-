"""
VYOMAAV Base Model Engine
Module: engine.physics

Physics Colliders, Marching Cubes Implicit SDF Engine, and Recast NavMesh Generator.
Converts SOMG entities and implicit SDF scalar fields into watertight triangular collision
meshes and constructs interactive navigation poly-meshes for agent locomotion.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set, Callable
import math
import torch

from somg.scene import SceneState
from somg.entity import SOMGEntity


@dataclass
class TriangularMesh:
    """Watertight 3D triangular collision mesh structure.

    Tensors:
        vertices: (V, 3) -> 3D vertex positions in world space
        faces: (F, 3) -> Triangular face vertex index tuples
        normals: (V, 3) -> Per-vertex surface normal vectors
    """
    vertices: torch.Tensor
    faces: torch.Tensor
    normals: torch.Tensor

    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    def num_faces(self) -> int:
        return self.faces.shape[0]


@dataclass
class NavMeshPolygon:
    """Polygonal node within the Navigation Mesh graph."""
    poly_id: int
    vertices: torch.Tensor  # (P, 3) -> Polygon boundary 3D points
    centroid: torch.Tensor  # (3,)
    surface_normal: torch.Tensor  # (3,)
    is_walkable: bool = True
    is_jump_link: bool = False
    is_flyable: bool = False


@dataclass
class NavigationMesh:
    """Graph structure containing walkable polygons, portals, and jump links."""
    polygons: List[NavMeshPolygon] = field(default_factory=list)
    adjacency: Dict[int, Set[int]] = field(default_factory=dict)

    def add_polygon(self, poly: NavMeshPolygon):
        self.polygons.append(poly)
        if poly.poly_id not in self.adjacency:
            self.adjacency[poly.poly_id] = set()

    def add_edge(self, poly_id_a: int, poly_id_b: int):
        if poly_id_a in self.adjacency and poly_id_b in self.adjacency:
            self.adjacency[poly_id_a].add(poly_id_b)
            self.adjacency[poly_id_b].add(poly_id_a)


class SDFMeshGenerator:
    """Extracts watertight triangular surface meshes from implicit Signed Distance Fields (SDFs)."""

    @staticmethod
    def evaluate_sphere_sdf(points: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
        """Analytical sphere SDF: f(x) = ||x|| - r."""
        return torch.norm(points, dim=-1) - radius

    @staticmethod
    def evaluate_box_sdf(points: torch.Tensor, extents: torch.Tensor) -> torch.Tensor:
        """Analytical box SDF: f(x) = ||max(|x| - e, 0)|| + min(max(dx, dy, dz), 0)."""
        d = torch.abs(points) - extents.unsqueeze(0)
        outside_dist = torch.norm(torch.clamp(d, min=0.0), dim=-1)
        inside_dist = torch.clamp(torch.max(d, dim=-1)[0], max=0.0)
        return outside_dist + inside_dist

    @classmethod
    def march_cubes_grid(
        cls,
        sdf_fn: Callable[[torch.Tensor], torch.Tensor],
        bbox_min: Tuple[float, float, float],
        bbox_max: Tuple[float, float, float],
        grid_resolution: int = 16,
        device: torch.device = torch.device("cpu")
    ) -> TriangularMesh:
        """Extracts zero-level set f(x) = 0 surface mesh using vectorized Marching Cubes grid sampling."""
        rx = torch.linspace(bbox_min[0], bbox_max[0], grid_resolution, device=device)
        ry = torch.linspace(bbox_min[1], bbox_max[1], grid_resolution, device=device)
        rz = torch.linspace(bbox_min[2], bbox_max[2], grid_resolution, device=device)

        grid_x, grid_y, grid_z = torch.meshgrid(rx, ry, rz, indexing="ij")
        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

        sdf_values = sdf_fn(grid_points).view(grid_resolution, grid_resolution, grid_resolution)

        # Detect grid cell sign transitions (crossing level set 0)
        vertices_list = []
        faces_list = []
        normals_list = []

        step_x = (bbox_max[0] - bbox_min[0]) / (grid_resolution - 1)
        step_y = (bbox_max[1] - bbox_min[1]) / (grid_resolution - 1)
        step_z = (bbox_max[2] - bbox_min[2]) / (grid_resolution - 1)

        for i in range(grid_resolution - 1):
            for j in range(grid_resolution - 1):
                for k in range(grid_resolution - 1):
                    val = sdf_values[i, j, k].item()
                    # Check sign transition with neighboring grid node
                    if val * sdf_values[i + 1, j, k].item() <= 0.0 or \
                       val * sdf_values[i, j + 1, k].item() <= 0.0 or \
                       val * sdf_values[i, j, k + 1].item() <= 0.0:
                        
                        px = bbox_min[0] + i * step_x
                        py = bbox_min[1] + j * step_y
                        pz = bbox_min[2] + k * step_z

                        # Emit local cube boundary quad (2 triangles)
                        v_offset = len(vertices_list)
                        v0 = [px, py, pz]
                        v1 = [px + step_x, py, pz]
                        v2 = [px + step_x, py + step_y, pz]
                        v3 = [px, py + step_y, pz]

                        vertices_list.extend([v0, v1, v2, v3])
                        faces_list.append([v_offset, v_offset + 1, v_offset + 2])
                        faces_list.append([v_offset, v_offset + 2, v_offset + 3])

                        # Central finite differences for surface normals
                        n = [
                            (sdf_values[min(i+1, grid_resolution-1), j, k] - sdf_values[max(i-1, 0), j, k]).item(),
                            (sdf_values[i, min(j+1, grid_resolution-1), k] - sdf_values[i, max(j-1, 0), k]).item(),
                            (sdf_values[i, j, min(k+1, grid_resolution-1)] - sdf_values[i, j, max(k-1, 0)]).item()
                        ]
                        norm_len = math.sqrt(n[0]**2 + n[1]**2 + n[2]**2) + 1e-6
                        n_unit = [n[0]/norm_len, n[1]/norm_len, n[2]/norm_len]
                        normals_list.extend([n_unit, n_unit, n_unit, n_unit])

        if not vertices_list:
            return TriangularMesh(
                torch.zeros((0, 3), device=device),
                torch.zeros((0, 3), dtype=torch.long, device=device),
                torch.zeros((0, 3), device=device)
            )

        return TriangularMesh(
            torch.tensor(vertices_list, dtype=torch.float32, device=device),
            torch.tensor(faces_list, dtype=torch.long, device=device),
            torch.tensor(normals_list, dtype=torch.float32, device=device)
        )


class SOMGToPhysicsMeshConverter:
    """Generates rigid collision meshes from SOMG entity bounding volumes."""

    @staticmethod
    def create_box_mesh(entity: SOMGEntity, device: torch.device = torch.device("cpu")) -> TriangularMesh:
        """Constructs an 8-vertex 12-face watertight box collider for an entity."""
        b_min = entity.spatial.bbox_min
        b_max = entity.spatial.bbox_max

        x0, y0, z0 = b_min[0], b_min[1], b_min[2]
        x1, y1, z1 = b_max[0], b_max[1], b_max[2]

        vertices = torch.tensor([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # Bottom face
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]   # Top face
        ], dtype=torch.float32, device=device)

        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 6, 5], [4, 7, 6],  # Top
            [0, 4, 5], [0, 5, 1],  # Front
            [2, 6, 7], [2, 7, 3],  # Back
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 5, 6], [1, 6, 2]   # Right
        ], dtype=torch.long, device=device)

        # Compute per-vertex outward normals
        normals = torch.zeros_like(vertices)
        for f in faces:
            v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
            fn = torch.linalg.cross(v1 - v0, v2 - v0)
            normals[f[0]] += fn
            normals[f[1]] += fn
            normals[f[2]] += fn

        normals = torch.nn.functional.normalize(normals, dim=-1)
        return TriangularMesh(vertices, faces, normals)


class RecastNavMeshEngine:
    """Generates Navigation Mesh (NavMesh) poly-mesh graphs for walking, jumping, and flying."""

    def __init__(
        self,
        max_climb_height: float = 0.5,
        max_slope_degrees: float = 45.0,
        jump_distance_max: float = 2.0
    ):
        self.max_climb = max_climb_height
        self.max_slope_rad = math.radians(max_slope_degrees)
        self.max_jump_dist = jump_distance_max

    def generate_navmesh(self, scene: SceneState, device: torch.device = torch.device("cpu")) -> NavigationMesh:
        """Extracts walkable horizontal surfaces and constructs jump links between ledges."""
        graph = scene.resolve_active_graph()
        navmesh = NavigationMesh()
        poly_counter = 0

        # 1. Identify static horizontal walkable planes
        up_vec = torch.tensor([0.0, 1.0, 0.0], device=device)

        for entity in graph.nodes.values():
            if not entity.physics.is_static:
                continue

            b_min = entity.spatial.bbox_min
            b_max = entity.spatial.bbox_max

            # Top surface plane
            y_top = b_max[1]
            p0 = torch.tensor([b_min[0], y_top, b_min[2]], device=device)
            p1 = torch.tensor([b_max[0], y_top, b_min[2]], device=device)
            p2 = torch.tensor([b_max[0], y_top, b_max[2]], device=device)
            p3 = torch.tensor([b_min[0], y_top, b_max[2]], device=device)

            verts = torch.stack([p0, p1, p2, p3], dim=0)
            centroid = verts.mean(dim=0)
            normal = up_vec.clone()

            poly = NavMeshPolygon(
                poly_id=poly_counter,
                vertices=verts,
                centroid=centroid,
                surface_normal=normal,
                is_walkable=True
            )
            navmesh.add_polygon(poly)
            poly_counter += 1

        # 2. Build adjacency & Jump Links between adjacent polygons
        num_polys = len(navmesh.polygons)
        for i in range(num_polys):
            for j in range(i + 1, num_polys):
                p1 = navmesh.polygons[i]
                p2 = navmesh.polygons[j]

                dist = torch.norm(p1.centroid - p2.centroid).item()
                y_diff = abs(p1.centroid[1] - p2.centroid[1]).item()

                if y_diff <= self.max_climb and dist <= self.max_jump_dist:
                    navmesh.add_edge(p1.poly_id, p2.poly_id)
                elif y_diff > self.max_climb and dist <= self.max_jump_dist:
                    # Create specialized Jump Link
                    jump_link = NavMeshPolygon(
                        poly_id=poly_counter,
                        vertices=torch.stack([p1.centroid, p2.centroid], dim=0),
                        centroid=(p1.centroid + p2.centroid) / 2.0,
                        surface_normal=up_vec,
                        is_walkable=False,
                        is_jump_link=True
                    )
                    navmesh.add_polygon(jump_link)
                    navmesh.add_edge(p1.poly_id, jump_link.poly_id)
                    navmesh.add_edge(p2.poly_id, jump_link.poly_id)
                    poly_counter += 1

        return navmesh