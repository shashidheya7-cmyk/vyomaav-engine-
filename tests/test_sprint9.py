"""
VYOMAAV Base Model Engine
Test Suite: tests/test_sprint9.py

Pytest suite validating Sprint 9: Gated Read/Write memory persistence,
pairwise relationship prediction tensor shapes, and graph edge writing.
"""

import pytest
import torch
from base_model.read_write_memory import ReadWriteWorldMemory
from base_model.relationship_head import NeuralRelationshipHead
from base_model.vision_encoder import VYOMAAVVisionEncoder
from base_model.camera_estimator import LieAlgebraCameraEstimator
from base_model.model import VYOMAAVBaseModel
from base_model.contracts import ModelInputBatch
from inference.pipeline_v2 import SpatialGraphInferencePipeline
from somg.scene import SceneState


def test_gated_read_write_memory_persistence():
    memory = ReadWriteWorldMemory(embed_dim=64, memory_slots=16, num_heads=4)
    
    tokens = torch.randn(2, 2, 16, 64)
    poses = torch.randn(2, 2, 3, 4)

    # Step 1: Initial Write
    state_t1 = memory(tokens, poses, prev_memory_state=None)
    assert state_t1.shape == (2, 16, 64)

    # Step 2: Second Write conditioned on state_t1
    state_t2 = memory(tokens, poses, prev_memory_state=state_t1)
    assert state_t2.shape == (2, 16, 64)
    assert not torch.allclose(state_t1, state_t2)


def test_neural_relationship_head_tensor_shapes():
    rel_head = NeuralRelationshipHead(embed_dim=64, num_relations=5)
    world_latent = torch.randn(2, 16, 64)  # (B=2, M=16 slots, D=64)

    rel_logits = rel_head(world_latent)

    assert rel_logits.shape == (2, 16, 16, 5)  # (B, M, M, num_relations)


def test_full_spatial_graph_inference_pipeline():
    vision = VYOMAAVVisionEncoder(embed_dim=64, patch_size=16)
    camera = LieAlgebraCameraEstimator(embed_dim=64)
    memory = ReadWriteWorldMemory(embed_dim=64, memory_slots=16)

    model = VYOMAAVBaseModel(
        embed_dim=64,
        num_classes=5,
        vision_encoder=vision,
        camera_estimator=camera,
        world_memory=memory
    )
    rel_head = NeuralRelationshipHead(embed_dim=64, num_relations=5)

    pipeline = SpatialGraphInferencePipeline(
        model=model,
        relationship_head=rel_head,
        confidence_threshold=0.1,
        rel_threshold=0.1,
        device="cpu"
    )

    batch = ModelInputBatch(frames=torch.randn(1, 2, 3, 32, 32))
    scene = SceneState(scene_id="RelationalWorld")

    delta = pipeline.run_inference_and_build_graph(batch, target_scene=scene)

    active_graph = scene.resolve_active_graph()
    assert len(active_graph.nodes) > 0