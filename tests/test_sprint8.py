"""
VYOMAAV Base Model Engine
Test Suite: tests/test_sprint8.py

Pytest suite validating Sprint 8: Cross-attention world memory tensor dimensions,
neural output decoding, end-to-end inference handoff, and SOMG graph updates.
"""

import pytest
import torch
from base_model.memory import AttentionWorldMemory
from base_model.vision_encoder import VYOMAAVVisionEncoder
from base_model.camera_estimator import LieAlgebraCameraEstimator
from base_model.model import VYOMAAVBaseModel
from base_model.contracts import ModelInputBatch
from inference.pipeline import BaseModelToSOMGInferencePipeline
from somg.scene import SceneState


def test_attention_world_memory_shape():
    memory = AttentionWorldMemory(embed_dim=64, memory_slots=16, num_heads=4)
    visual_tokens = torch.randn(2, 3, 16, 64)    # (B=2, T=3, N=16, D=64)
    camera_poses = torch.randn(2, 3, 3, 4)     # (B=2, T=3, 3, 4)

    world_latent = memory(visual_tokens, camera_poses)

    assert world_latent.shape == (2, 16, 64)  # (B, M=16 slots, D=64)
    assert not torch.isnan(world_latent).any()


def test_end_to_end_neural_to_somg_inference_handoff():
    # Build model with real vision, camera, and attention memory modules
    vision = VYOMAAVVisionEncoder(embed_dim=64, patch_size=16, num_layers=2)
    camera = LieAlgebraCameraEstimator(embed_dim=64)
    memory = AttentionWorldMemory(embed_dim=64, memory_slots=16, num_heads=4)

    model = VYOMAAVBaseModel(
        embed_dim=64,
        num_classes=5,
        vision_encoder=vision,
        camera_estimator=camera,
        world_memory=memory
    )

    class_map = {0: "chair", 1: "table", 2: "lamp"}
    pipeline = BaseModelToSOMGInferencePipeline(
        model=model, class_id_to_label=class_map, confidence_threshold=0.1, device="cpu"
    )

    # Input batch
    dummy_frames = torch.randn(1, 2, 3, 32, 32)
    batch = ModelInputBatch(frames=dummy_frames)

    # Initialize target SOMG scene state
    scene = SceneState(scene_id="InferenceWorld")

    # Run inference and update SOMG
    delta = pipeline.run_inference_and_update_somg(batch, target_scene=scene, delta_id="frame_delta_01")

    # Verify SOMG state update
    active_graph = scene.resolve_active_graph()
    assert len(active_graph.nodes) > 0
    
    first_node_id = list(active_graph.nodes.keys())[0]
    first_node = active_graph.nodes[first_node_id]

    assert first_node.semantic.label in class_map.values() or "class_" in first_node.semantic.label
    assert first_node.spatial.bbox_min is not None
    assert first_node.spatial.bbox_max is not None