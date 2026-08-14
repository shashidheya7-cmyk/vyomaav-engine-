from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

@dataclass
class SegmentationMask:
    mask_id: str
    object_id: str
    mask_array: np.ndarray  # bool or uint8 HxW
    confidence: float
    bbox: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask_id": self.mask_id,
            "object_id": self.object_id,
            "mask_shape": list(self.mask_array.shape),
            "confidence": self.confidence,
            "bbox": self.bbox,
            "metadata": self.metadata
        }

@dataclass
class SegmentationSet:
    observation_id: str
    masks: List[SegmentationMask] = field(default_factory=list)
    tracked_ids: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "masks": [m.to_dict() for m in self.masks],
            "tracked_ids": self.tracked_ids,
            "provenance": self.provenance
        }
