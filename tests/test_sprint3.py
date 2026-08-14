"""
VYOMAAV Base Model Engine
Test Suite: tests/test_sprint3.py

Pytest suite validating Sprint 3: Perception conversion, 3D IoU computation,
temporal observation fusion, Bayesian memory updates, and OpenUSD/JSON-LD exports.
"""

import pytest
import json
from somg.scene import SceneState
from somg.builder import PerceptionObservation, SOMGEntityBuilder
from somg.fusion import TemporalFusionEngine
from somg.exporter import SOMGExporter


def test_3d_iou_computation():
    box1_min = [0.0, 0.0, 0.0]
    box1_max = [2.0, 2.0, 2.0]

    box2_min = [1.0, 1.0, 1.0]
    box2_max = [3.0, 3.0, 3.0]

    # Intersection is [1, 1, 1] to [2, 2, 2] -> Volume = 1.0
    # Box1 vol = 8.0, Box2 vol = 8.0 -> Union = 15.0 -> IoU = 1 / 15 = ~0.0667
    iou = TemporalFusionEngine.compute_3d_iou(box1_min, box1_max, box2_min, box2_max)
    assert 0.06 < iou < 0.07

    # Non-overlapping
    box3_min = [5.0, 5.0, 5.0]
    box3_max = [6.0, 6.0, 6.0]
    iou_zero = TemporalFusionEngine.compute_3d_iou(box1_min, box1_max, box3_min, box3_max)
    assert iou_zero == 0.0


def test_temporal_fusion_deduplication_and_reconciliation():
    scene = SceneState(scene_id="Office")
    fusion = TemporalFusionEngine(scene, iou_threshold=0.2)

    # Frame 1: Initial Detections
    obs_f1 = [
        PerceptionObservation(
            observation_id="001", label="table", class_id=1, confidence=0.80,
            bbox_min=[-1.0, 0.0, -1.0], bbox_max=[1.0, 1.0, 1.0]
        )
    ]
    fusion.fuse_frame_observations(obs_f1, delta_id="frame_1")
    
    g1 = scene.resolve_active_graph()
    assert len(g1.nodes) == 1
    table_id = list(g1.nodes.keys())[0]
    assert g1.nodes[table_id].semantic.confidence == 0.80

    # Frame 2: Re-observing the same table (overlapping BBox)
    obs_f2 = [
        PerceptionObservation(
            observation_id="002", label="table", class_id=1, confidence=0.85,
            bbox_min=[-0.95, 0.02, -0.95], bbox_max=[1.05, 1.02, 1.05]
        )
    ]
    fusion.fuse_frame_observations(obs_f2, delta_id="frame_2")

    g2 = scene.resolve_active_graph()
    assert len(g2.nodes) == 1  # Deduplicated into 1 persistent entity
    assert g2.nodes[table_id].version == 2
    assert g2.nodes[table_id].semantic.confidence > 0.80  # Bayesian confidence boost


def test_exporter_jsonld_and_openusd():
    scene = SceneState(scene_id="LabScene")
    obs = PerceptionObservation(
        observation_id="99", label="robot", class_id=10, confidence=0.99,
        bbox_min=[0.0, 0.0, 0.0], bbox_max=[1.0, 2.0, 1.0], estimated_mass_kg=45.0
    )
    e = SOMGEntityBuilder.from_observation(obs)
    scene.base_graph.add_node(e)

    # JSON-LD Export
    jsonld_str = SOMGExporter.to_json_ld(scene)
    data = json.loads(jsonld_str)
    assert data["scene_id"] == "LabScene"
    assert data["entity_count"] == 1
    assert data["entities"][0]["semantic"]["label"] == "robot"

    # OpenUSD ASCII Export
    usd_str = SOMGExporter.to_openusd_ascii(scene)
    assert '#usda 1.0' in usd_str
    assert 'def Xform "LabScene"' in usd_str
    assert 'custom string vyomaav:label = "robot"' in usd_str