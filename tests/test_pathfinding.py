"""
VYOMAAV Base Model Engine
Test Suite: tests/test_pathfinding.py

Pytest suite validating Sprint 15: A* NavMesh pathfinding, Euclidean distance heuristics,
jump-link penalty evaluation, and 3D waypoint trajectory extraction.
"""

import pytest
import torch
from somg.scene import SceneState
from somg.entity import SOMGEntity, SpatialComponent, PhysicsComponent
from engine.physics import RecastNavMeshEngine, NavigationMesh
from engine.pathfinding import AStarNavMeshPathfinder, PathfindingResult


def test_astar_pathfinding_connected_platforms():
    scene = SceneState(scene_id="PathScene")

    # Platform 1
    p1 = SOMGEntity("p1", spatial=SpatialComponent(bbox_min=[0.0, 0.0, 0.0], bbox_max=[2.0, 0.2, 2.0]), physics=PhysicsComponent(is_static=True))
    # Platform 2 (adjacent)
    p2 = SOMGEntity("p2", spatial=SpatialComponent(bbox_min=[2.2, 0.0, 0.0], bbox_max=[4.2, 0.2, 2.0]), physics=PhysicsComponent(is_static=True))

    scene.base_graph.add_node(p1)
    scene.base_graph.add_node(p2)

    engine = RecastNavMeshEngine(max_climb_height=0.5, jump_distance_max=1.0)
    navmesh = engine.generate_navmesh(scene, device=torch.device("cpu"))

    pathfinder = AStarNavMeshPathfinder(navmesh)
    res = pathfinder.find_path(start_pos=[0.5, 0.2, 0.5], target_pos=[3.5, 0.2, 0.5])

    assert res.found is True
    assert res.total_cost > 0.0
    assert len(res.path_poly_ids) >= 2
    assert len(res.waypoints) >= 2
    assert res.waypoints[0].position == [0.5, 0.2, 0.5]


def test_astar_pathfinding_across_jump_link():
    scene = SceneState(scene_id="JumpScene")

    # Lower platform
    floor = SOMGEntity("floor", spatial=SpatialComponent(bbox_min=[-2.0, 0.0, -2.0], bbox_max=[0.0, 0.2, 2.0]), physics=PhysicsComponent(is_static=True))
    # Higher elevated ledge
    ledge = SOMGEntity("ledge", spatial=SpatialComponent(bbox_min=[1.0, 1.2, -1.0], bbox_max=[3.0, 1.4, 1.0]), physics=PhysicsComponent(is_static=True))

    scene.base_graph.add_node(floor)
    scene.base_graph.add_node(ledge)

    navmesh = RecastNavMeshEngine(max_climb_height=0.5, jump_distance_max=3.0).generate_navmesh(scene)

    pathfinder = AStarNavMeshPathfinder(navmesh, jump_cost_penalty=3.0)
    res = pathfinder.find_path(start_pos=[-1.0, 0.2, 0.0], target_pos=[2.0, 1.4, 0.0])

    assert res.found is True
    assert any(wp.is_jump for wp in res.waypoints)


def test_astar_pathfinding_unreachable_target():
    scene = SceneState(scene_id="IsolatedScene")
    p1 = SOMGEntity("p1", spatial=SpatialComponent(bbox_min=[0.0, 0.0, 0.0], bbox_max=[1.0, 0.2, 1.0]), physics=PhysicsComponent(is_static=True))
    # Isolated platform far away
    p2 = SOMGEntity("p2", spatial=SpatialComponent(bbox_min=[100.0, 0.0, 0.0], bbox_max=[101.0, 0.2, 1.0]), physics=PhysicsComponent(is_static=True))

    scene.base_graph.add_node(p1)
    scene.base_graph.add_node(p2)

    navmesh = RecastNavMeshEngine(jump_distance_max=2.0).generate_navmesh(scene)

    pathfinder = AStarNavMeshPathfinder(navmesh)
    res = pathfinder.find_path(start_pos=[0.5, 0.2, 0.5], target_pos=[100.5, 0.2, 0.5])

    assert res.found is False
    assert res.total_cost == float("inf")