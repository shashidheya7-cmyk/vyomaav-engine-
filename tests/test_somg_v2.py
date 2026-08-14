"""
VYOMAAV Base Model Engine
Test Suite: tests/test_somg_v2.py

Pytest suite validating Sprint 2.2: Spatial Indexing, Component Preservation, Queries, and Delta Optimization.
"""

import pytest
from somg.entity import SOMGEntity, SemanticComponent, SpatialComponent, PhysicsComponent
from somg.scene import SceneState, DeltaLayer
from somg.spatial_index import AABBSpatialIndex
from somg.query import SOMGQueryEngine
from somg.optimizer import DeltaOptimizer
from somg.graph import RelationType


def test_component_preservation_and_versioning():
    entity = SOMGEntity(
        entity_id="chair_01",
        version=1,
        semantic=SemanticComponent(label="armchair", class_id=102),
        spatial=SpatialComponent(bbox_min=[-0.5, 0.0, -0.5], bbox_max=[0.5, 1.0, 0.5]),
        physics=PhysicsComponent(mass_kg=12.0)
    )

    v2_entity = entity.increment_version()

    assert entity.version == 1
    assert v2_entity.version == 2
    assert v2_entity.semantic.label == "armchair"
    assert v2_entity.physics.mass_kg == 12.0


def test_spatial_index_3d_bounding_box_overlap():
    index = AABBSpatialIndex(cell_size=2.0)
    index.insert_or_update("box_A", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    index.insert_or_update("box_B", [5.0, 5.0, 5.0], [6.0, 6.0, 6.0])

    # Query overlapping box_A only
    results_A = index.query_aabb_overlap([-0.5, -0.5, -0.5], [1.5, 1.5, 1.5])
    assert "box_A" in results_A
    assert "box_B" not in results_A

    # Query bounding volume encompassing both
    results_both = index.query_aabb_overlap([-1.0, -1.0, -1.0], [10.0, 10.0, 10.0])
    assert "box_A" in results_both
    assert "box_B" in results_both


def test_query_engine_spatial_and_bfs_traversal():
    scene = SceneState(scene_id="Room")
    
    floor = SOMGEntity(entity_id="floor", semantic=SemanticComponent(label="floor"), spatial=SpatialComponent(bbox_min=[-5.0, -0.1, -5.0], bbox_max=[5.0, 0.0, 5.0]))
    table = SOMGEntity(entity_id="table", semantic=SemanticComponent(label="table"), spatial=SpatialComponent(bbox_min=[-1.0, 0.0, -1.0], bbox_max=[1.0, 1.0, 1.0]))
    cup = SOMGEntity(entity_id="cup", semantic=SemanticComponent(label="cup"), spatial=SpatialComponent(bbox_min=[-0.1, 1.0, -0.1], bbox_max=[0.1, 1.2, 0.1]))

    scene.base_graph.add_node(floor)
    scene.base_graph.add_node(table)
    scene.base_graph.add_node(cup)

    scene.base_graph.add_edge("table", "floor", RelationType.SUPPORTED_BY)
    scene.base_graph.add_edge("cup", "table", RelationType.SUPPORTED_BY)

    query = SOMGQueryEngine(scene)

    # Test spatial volume query
    found_spatial = query.find_in_volume([-1.5, -0.5, -1.5], [1.5, 1.5, 1.5])
    found_ids = {e.entity_id for e in found_spatial}
    assert "floor" in found_ids
    assert "table" in found_ids
    assert "cup" in found_ids

    # Test BFS graph traversal from table
    bfs_results = query.bfs_traversal("table", max_depth=2)
    assert len(bfs_results) == 1
    assert bfs_results[0][0].entity_id == "floor"


def test_delta_squashing_and_checkpointing():
    scene = SceneState(scene_id="BaseWorld")
    scene.base_graph.add_node(SOMGEntity(entity_id="e1", semantic=SemanticComponent(label="e1")))

    # Delta 1: Add e2
    d1 = DeltaLayer(layer_id="d1")
    d1.added_entities["e2"] = SOMGEntity(entity_id="e2", semantic=SemanticComponent(label="e2"))
    
    # Delta 2: Add e3, remove e1
    d2 = DeltaLayer(layer_id="d2")
    d2.added_entities["e3"] = SOMGEntity(entity_id="e3", semantic=SemanticComponent(label="e3"))
    d2.removed_entity_ids.add("e1")

    squashed_delta = DeltaOptimizer.squash_deltas([d1, d2], new_layer_id="squashed_d1_d2")

    assert "e2" in squashed_delta.added_entities
    assert "e3" in squashed_delta.added_entities
    assert "e1" in squashed_delta.removed_entity_ids

    # Checkpoint
    scene.push_delta(d1)
    scene.push_delta(d2)
    checkpoint_scene = DeltaOptimizer.create_checkpoint(scene, "v1")

    active_ckpt_graph = checkpoint_scene.resolve_active_graph()
    assert "e1" not in active_ckpt_graph.nodes
    assert "e2" in active_ckpt_graph.nodes
    assert "e3" in active_ckpt_graph.nodes