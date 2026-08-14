"""Bounding volume representations for spatial pruning and collision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass
class BoundingBox3D:
    """Axis-Aligned 3D Bounding Box."""

    min_point: List[float] = field(default_factory=lambda: [-1.0, -1.0, -1.0])
    max_point: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    @property
    def center(self) -> List[float]:
        return [0.5 * (mn + mx) for mn, mx in zip(self.min_point, self.max_point)]

    @property
    def extents(self) -> List[float]:
        return [mx - mn for mn, mx in zip(self.min_point, self.max_point)]

    def intersects(self, other: BoundingBox3D) -> bool:
        """Check if this box intersects another box."""
        for i in range(3):
            if self.max_point[i] < other.min_point[i] or self.min_point[i] > other.max_point[i]:
                return False
        return True

    def union(self, other: BoundingBox3D) -> BoundingBox3D:
        """Return the minimal bounding box containing both boxes."""
        new_min = [min(a, b) for a, b in zip(self.min_point, other.min_point)]
        new_max = [max(a, b) for a, b in zip(self.max_point, other.max_point)]
        return BoundingBox3D(min_point=new_min, max_point=new_max)

    def to_dict(self) -> Dict[str, Any]:
        return {"min_point": self.min_point, "max_point": self.max_point}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BoundingBox3D:
        return cls(**data)
