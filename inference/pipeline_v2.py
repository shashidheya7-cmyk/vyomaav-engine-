"""
VYOMAAV Base Model Engine
Module: inference.pipeline_v2

Extended Inference Pipeline that decodes spatial entity attributes AND neural relationship
logits, writing directed spatial graph edges directly into the SOMG SceneState.
"""

from typing import List, Dict, Tuple, Optional
import torch
from base_model.model import VYOMAAVBaseModel
from base_model.relationship_head import NeuralRelationshipHead
from base_model.contracts import ModelInputBatch, ModelOutputState
from somg.builder import PerceptionObservation
from somg.fusion import TemporalFusionEngine
from somg.scene import SceneState, DeltaLayer
from somg.graph import RelationType


class SpatialGraphInferencePipeline:
    """Executes full neural inference and writes entities AND spatial graph edges to SOMG."""

    RELATION_MAP = {
        0: RelationType.SUPPORTED_BY,
        1: RelationType.CONTAINS,
        2: RelationType.ADJACENT_TO,
        3: RelationType.BLOCKS_PATH,
        4: RelationType.ATTACHED_TO
    }

    def __init__(
        self,
        model: VYOMAAVBaseModel,
        relationship_head: NeuralRelationshipHead,
        class_id_to_label: Optional[Dict[int, str]] = None,
        confidence_threshold: float = 0.5,
        rel_threshold: float = 0.6,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.rel_head = relationship_head.to(self.device)
        self.model.eval()
        self.rel_head.eval()

        self.confidence_threshold = confidence_threshold
        self.rel_threshold = rel_threshold
        self.class_id_to_label = class_id_to_label or {i: f"class_{i}" for i in range(100)}

    def run_inference_and_build_graph(
        self,
        batch: ModelInputBatch,
        target_scene: SceneState,
        delta_id: str = "neural_graph_update"
    ) -> DeltaLayer:
        batch.frames = batch.frames.to(self.device)
        if batch.depth_maps is not None:
            batch.depth_maps = batch.depth_maps.to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            rel_logits = self.rel_head(outputs.world_latent)  # (B, M, M, num_relations)

        # 1. Decode entities
        logits = outputs.entity_class_logits[0]  # Batch item 0
        bboxes = outputs.entity_bboxes[0]
        probs = torch.softmax(logits, dim=-1)
        confidences, class_ids = torch.max(probs, dim=-1)

        observations: List[PerceptionObservation] = []
        valid_slot_to_obs_id: Dict[int, str] = {}

        for i in range(logits.shape[0]):
            conf = confidences[i].item()
            if conf < self.confidence_threshold:
                continue

            cls_id = class_ids[i].item()
            label = self.class_id_to_label.get(cls_id, f"class_{cls_id}")
            bbox_raw = bboxes[i].tolist()

            bbox_min = [min(bbox_raw[0], bbox_raw[3]), min(bbox_raw[1], bbox_raw[4]), min(bbox_raw[2], bbox_raw[5])]
            bbox_max = [max(bbox_raw[0], bbox_raw[3]), max(bbox_raw[1], bbox_raw[4]), max(bbox_raw[2], bbox_raw[5])]

            obs_id = f"slot_{i}"
            valid_slot_to_obs_id[i] = obs_id

            obs = PerceptionObservation(
                observation_id=f"obs_{i}",
                label=label,
                class_id=cls_id,
                confidence=conf,
                bbox_min=bbox_min,
                bbox_max=bbox_max
            )
            observations.append(obs)

        # 2. Fuse entities into SOMG DeltaLayer
        fusion = TemporalFusionEngine(target_scene, iou_threshold=0.2)
        delta = fusion.fuse_frame_observations(observations, delta_id=delta_id)

        # 3. Decode relationship edges and write to SOMG base graph
        rel_probs = torch.sigmoid(rel_logits[0])  # (M, M, num_relations)
        active_graph = target_scene.resolve_active_graph()

        for i in valid_slot_to_obs_id:
            for j in valid_slot_to_obs_id:
                if i == j:
                    continue  # No self-loops

                src_id = valid_slot_to_obs_id[i]
                tgt_id = valid_slot_to_obs_id[j]

                for rel_idx, rel_enum in self.RELATION_MAP.items():
                    prob = rel_probs[i, j, rel_idx].item()
                    if prob >= self.rel_threshold:
                        # Write directed relational edge to scene graph
                        target_scene.base_graph.add_edge(
                            source_id=src_id, target_id=tgt_id, relation_type=rel_enum
                        )

        return delta