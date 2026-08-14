"""VYOMAAV Base Model Data Contracts."""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import torch

@dataclass
class ModelInputBatch:
    frames: Optional[torch.Tensor] = None
    rgb_images: Optional[torch.Tensor] = None
    depth_images: Optional[torch.Tensor] = None
    depth_maps: Optional[torch.Tensor] = None
    intrinsics: Optional[torch.Tensor] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.frames is not None and self.rgb_images is None:
            self.rgb_images = self.frames
        if self.depth_maps is not None and self.depth_images is None:
            self.depth_images = self.depth_maps
        elif self.depth_images is not None and self.depth_maps is None:
            self.depth_maps = self.depth_images

@dataclass
class ModelOutputState:
    predicted_bboxes: Optional[torch.Tensor] = field(default_factory=lambda: torch.zeros((1, 2, 6)))
    predicted_classes: Optional[torch.Tensor] = field(default_factory=lambda: torch.zeros((1, 2, 10)))
    predicted_relationships: Optional[torch.Tensor] = field(default_factory=lambda: torch.zeros((1, 2, 2, 5)))
    memory_state: Optional[torch.Tensor] = field(default_factory=lambda: torch.zeros((1, 16, 64)))
    pred_intrinsics_k: Optional[torch.Tensor] = field(default_factory=lambda: torch.zeros((1, 2, 3, 3)))
    pred_poses_se3: Optional[torch.Tensor] = field(default_factory=lambda: torch.zeros((1, 2, 3, 4)))
    entity_class_logits: Optional[torch.Tensor] = field(default_factory=lambda: torch.zeros((1, 2, 10)))
    entity_bboxes: Optional[torch.Tensor] = field(default_factory=lambda: torch.zeros((1, 2, 6)))
    entity_masses: Optional[torch.Tensor] = field(default_factory=lambda: torch.ones((1, 2, 1)))
    entity_uncertainty: Optional[torch.Tensor] = field(default_factory=lambda: torch.zeros((1, 2, 2)))
    world_latent: Optional[torch.Tensor] = field(default_factory=lambda: torch.zeros((1, 16, 64)))
    extra_outputs: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)
