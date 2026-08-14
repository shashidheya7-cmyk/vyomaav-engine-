"""
VYOMAAV Base Model Engine
Module: inference.pipeline

End-to-End Inference Handoff Pipeline. Executes neural forward passes on input frame
batches, decodes predicted bounding boxes and classes into PerceptionObservations,
and fuses them into a persistent SOMG SceneState using TemporalFusionEngine.
"""

from typing import List, Dict, Tuple, Optional
import torch
from base_model.model import VYOMAAVBaseModel
from base_model.contracts import ModelInputBatch, ModelOutputState
from somg.builder import PerceptionObservation
from somg.fusion import TemporalFusionEngine
from somg.scene import SceneState, DeltaLayer


class BaseModelToSOMGInferencePipeline:
    """Executes neural inference and bridges predictions directly into SOMG scene states."""

    def __init__(
        self,
        model: VYOMAAVBaseModel,
        class_id_to_label: Optional[Dict[int, str]] = None,
        confidence_threshold: float = 0.5,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        self.class_id_to_label = class_id_to_label or {i: f"object_class_{i}" for i in range(100)}

    def decode_output_to_observations(
        self, output: ModelOutputState, batch_idx: int = 0
    ) -> List[PerceptionObservation]:
        """Translates ModelOutputState tensor predictions into PerceptionObservation objects."""
        observations: List[PerceptionObservation] = []

        # Tensors for target batch item
        logits = output.entity_class_logits[batch_idx]  # (E, num_classes)
        bboxes = output.entity_bboxes[batch_idx]        # (E, 6)
        masses = torch.abs(output.entity_masses[batch_idx])  # (E, 1)
        uncs = torch.sigmoid(output.entity_uncertainty[batch_idx])  # (E, 2) -> [aleatoric, epistemic]

        probs = torch.softmax(logits, dim=-1)
        confidences, class_ids = torch.max(probs, dim=-1)

        for i in range(logits.shape[0]):
            conf = confidences[i].item()
            if conf < self.confidence_threshold:
                continue

            cls_id = class_ids[i].item()
            label = self.class_id_to_label.get(cls_id, f"class_{cls_id}")
            bbox_raw = bboxes[i].tolist()

            # Format 6D bbox tensor into min/max points
            bbox_min = [min(bbox_raw[0], bbox_raw[3]), min(bbox_raw[1], bbox_raw[4]), min(bbox_raw[2], bbox_raw[5])]
            bbox_max = [max(bbox_raw[0], bbox_raw[3]), max(bbox_raw[1], bbox_raw[4]), max(bbox_raw[2], bbox_raw[5])]

            obs = PerceptionObservation(
                observation_id=f"pred_slot_{i}",
                label=label,
                class_id=cls_id,
                confidence=conf,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                estimated_mass_kg=max(0.1, masses[i].item()),
                aleatoric_noise=uncs[i, 0].item()
            )
            observations.append(obs)

        return observations

    def run_inference_and_update_somg(
        self,
        batch: ModelInputBatch,
        target_scene: SceneState,
        delta_id: str = "neural_update"
    ) -> DeltaLayer:
        """Executes full pipeline: Tensors -> Neural Forward -> Observations -> SOMG Temporal Fusion."""
        # Move batch tensors to device
        batch.frames = batch.frames.to(self.device)
        if batch.depth_maps is not None:
            batch.depth_maps = batch.depth_maps.to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)

        # Decode batch item 0 into observations
        observations = self.decode_output_to_observations(outputs, batch_idx=0)

        # Fuse observations into target SOMG scene
        fusion = TemporalFusionEngine(target_scene, iou_threshold=0.2)
        delta = fusion.fuse_frame_observations(observations, delta_id=delta_id)

        return delta