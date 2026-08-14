"""
VYOMAAV Base Model Engine
Module: somg.fusion

Temporal Fusion & Reconciliation Engine. Merges multi-frame perception observations
into persistent SOMG scene states, performs 3D IoU spatial deduplication, updates
entity confidence, and resolves conflicts using non-destructive DeltaLayers.
"""

from typing import List, Dict, Tuple, Optional, Set
from somg.scene import SceneState, DeltaLayer
from somg.entity import SOMGEntity, SpatialComponent, SemanticComponent, UncertaintyComponent
from somg.builder import PerceptionObservation, SOMGEntityBuilder
from somg.spatial_index import AABBSpatialIndex


class TemporalFusionEngine:
    """Reconciles frame observations with persistent world memory."""

    def __init__(self, scene: SceneState, iou_threshold: float = 0.3):
        self.scene = scene
        self.iou_threshold = iou_threshold

    @staticmethod
    def compute_3d_iou(box1_min: List[float], box1_max: List[float],
                       box2_min: List[float], box2_max: List[float]) -> float:
        """Calculates 3D Axis-Aligned Bounding Box Intersection over Union (IoU)."""
        inter_min = [max(box1_min[i], box2_min[i]) for i in range(3)]
        inter_max = [min(box1_max[i], box2_max[i]) for i in range(3)]

        inter_dims = [max(0.0, inter_max[i] - inter_min[i]) for i in range(3)]
        inter_vol = inter_dims[0] * inter_dims[1] * inter_dims[2]

        vol1 = (box1_max[0] - box1_min[0]) * (box1_max[1] - box1_min[1]) * (box1_max[2] - box1_min[2])
        vol2 = (box2_max[0] - box2_min[0]) * (box2_max[1] - box2_min[1]) * (box2_max[2] - box2_min[2])

        union_vol = vol1 + vol2 - inter_vol
        if union_vol <= 0.0:
            return 0.0
        return inter_vol / union_vol

    def fuse_frame_observations(self, observations: List[PerceptionObservation], delta_id: str) -> DeltaLayer:
        """Fuses incoming frame detections into a new non-destructive DeltaLayer."""
        active_graph = self.scene.resolve_active_graph()
        spatial_index = AABBSpatialIndex(cell_size=2.0)

        # Index existing active scene entities
        for entity in active_graph.nodes.values():
            if entity.spatial:
                spatial_index.insert_or_update(
                    entity.entity_id, entity.spatial.bbox_min, entity.spatial.bbox_max
                )

        delta = DeltaLayer(layer_id=delta_id)
        matched_entity_ids: Set[str] = set()

        for obs in observations:
            candidates = spatial_index.query_aabb_overlap(obs.bbox_min, obs.bbox_max)
            best_match_id: Optional[str] = None
            best_iou = 0.0

            for cand_id in candidates:
                cand_entity = active_graph.nodes[cand_id]
                # Match only if semantic label matches
                if cand_entity.semantic.label == obs.label:
                    iou = self.compute_3d_iou(
                        obs.bbox_min, obs.bbox_max,
                        cand_entity.spatial.bbox_min, cand_entity.spatial.bbox_max
                    )
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_match_id = cand_id

            if best_match_id:
                # Deduplication / Reconciliation Pass (Update existing entity)
                existing = active_graph.nodes[best_match_id]
                matched_entity_ids.add(best_match_id)

                # Exponential Moving Average blending for bounding box refinement
                alpha = 0.3
                new_min = [alpha * obs.bbox_min[i] + (1 - alpha) * existing.spatial.bbox_min[i] for i in range(3)]
                new_max = [alpha * obs.bbox_max[i] + (1 - alpha) * existing.spatial.bbox_max[i] for i in range(3)]

                # Bayesian confidence boost upon re-observation
                updated_conf = min(1.0, existing.semantic.confidence + 0.05)

                updated_entity = SOMGEntity(
                    entity_id=existing.entity_id,
                    version=existing.version + 1,
                    semantic=SemanticComponent(
                        label=existing.semantic.label,
                        class_id=existing.semantic.class_id,
                        confidence=updated_conf
                    ),
                    spatial=SpatialComponent(
                        bbox_min=new_min,
                        bbox_max=new_max,
                        transform_matrix=existing.spatial.transform_matrix,
                        sdf_ref=existing.spatial.sdf_ref
                    ),
                    material=existing.material,
                    physics=existing.physics,
                    uncertainty=UncertaintyComponent(
                        aleatoric_noise=min(existing.uncertainty.aleatoric_noise, obs.aleatoric_noise),
                        epistemic_risk=max(0.0, existing.uncertainty.epistemic_risk - 0.05),
                        is_inferred=False
                    )
                )
                delta.updated_entities[existing.entity_id] = updated_entity
            else:
                # New Entity Insertion Pass
                new_entity = SOMGEntityBuilder.from_observation(obs)
                delta.added_entities[new_entity.entity_id] = new_entity

        self.scene.push_delta(delta)
        return delta