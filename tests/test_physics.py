"""
VYOMAAV Base Model Engine
Test Suite: tests/test_physics.py

Pytest suite validating Sprint 11: Box collision mesh generation, Marching Cubes implicit SDF
surface extraction, and Recast NavMesh poly-mesh construction with jump links.
"""

import pytest
import torch
from somg.scene import SceneState
from somg.entity import SOMGEntity, SpatialComponent, PhysicsComponent
from engine.physics import (
    TriangularMesh, SDFMeshGenerator, SOMGToPhysicsMeshConverter,
    RecastNavMeshEngine, NavigationMesh
)


def test_somg_box_collision_mesh_generation():
    entity = SOMGEntity(
        entity_id="obstacle_cube",
        spatial=SpatialComponent(bbox_min=[-1.0, -1.0, -1.0], bbox_max=[1.0, 1.0, 1.0])
    )

    mesh = SOMGToPhysicsMeshConverter.create_box_mesh(entity, device=torch.device("cpu"))

    assert isinstance(mesh, TriangularMesh)
    assert mesh.num_vertices() == 8
    assert mesh.num_faces() == 12  # 6 cube faces * 2 triangles
    assert mesh.normals.shape == (8, 3)


def test_marching_cubes_sdf_sphere_mesh_extraction():
    # Sphere SDF: radius 1.0 centered at origin
    sdf_fn = lambda pts: SDFMeshGenerator.evaluate_sphere_sdf(pts, radius=1.0)

    mesh = SDFMeshGenerator.march_cubes_grid(
        sdf_fn=sdf_fn,
        bbox_min=(-1.5, -1.5, -1.5),
        bbox_max=(1.5, 1.5, 1.5),
        grid_resolution=10,
        device=torch.device("cpu")
    )

    assert isinstance(mesh, TriangularMesh)
    assert mesh.num_vertices() > 0
    assert mesh.num_faces() > 0
    assert mesh.normals.shape[0] == mesh.num_vertices()


def test_recast_navmesh_walkable_polygons_and_jump_links():
    scene = SceneState(scene_id="PlatformRoom")

    # Lower Floor Platform
    floor = SOMGEntity(
        entity_id="floor",
        spatial=SpatialComponent(bbox_min=[-5.0, 0.0, -5.0], bbox_max=[5.0, 0.2, 5.0]),
        physics=PhysicsComponent(is_static=True)
    )

    # Elevated Ledge Platform
    ledge = SOMGEntity(
        entity_id="ledge",
        spatial=SpatialComponent(bbox_min=[1.0, 1.0, -1.0], bbox_max=[3.0, 1.2, 1.0]),
        physics=PhysicsComponent(is_static=True)
    )

    scene.base_graph.add_node(floor)
    scene.base_graph.add_node(ledge)

    engine = RecastNavMeshEngine(max_climb_height=0.5, jump_distance_max=4.0)
    navmesh = engine.generate_navmesh(scene, device=torch.device("cpu"))

    assert isinstance(navmesh, NavigationMesh)
    assert len(navmesh.polygons) >= 2  # Floor + Ledge + Jump Link
    
    # Verify presence of a Jump Link between platforms
    jump_links = [p for p in navmesh.polygons if p.is_jump_link]
    assert len(jump_links) == 1
    assert jump_links[0].poly_id in navmesh.adjacency