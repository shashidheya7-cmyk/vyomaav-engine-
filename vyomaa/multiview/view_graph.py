"""Data structures for multi-view graphs, pairwise matches, and quality scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..core.contracts import Camera, Observation


@dataclass
class CorrespondenceMap:
    """2D-2D pixel coordinate correspondences between two views."""
    view_a_id: str
    view_b_id: str
    points_a: np.ndarray  # (M, 2)
    points_b: np.ndarray  # (M, 2)
    match_scores: np.ndarray  # (M,)
    inlier_mask: np.ndarray   # (M,) boolean

    @property
    def num_matches(self) -> int:
        return len(self.points_a)

    @property
    def num_inliers(self) -> int:
        return int(np.sum(self.inlier_mask))

    @property
    def inlier_ratio(self) -> float:
        return float(self.num_inliers) / max(self.num_matches, 1)


@dataclass
class ViewQualityScore:
    """Individual view image quality metrics."""
    view_id: str
    sharpness: float
    exposure_balance: float
    feature_density: int
    overall_quality: float


@dataclass
class ViewPair:
    """Pairwise connection between two viewpoints in the multi-view graph."""
    view_a_id: str
    view_b_id: str
    correspondence: CorrespondenceMap
    relative_R: np.ndarray  # (3, 3)
    relative_t: np.ndarray  # (3,)
    epipolar_error_pixels: float
    overlap_score: float
    geometric_consistency_score: float
    is_valid_edge: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "view_a_id": self.view_a_id,
            "view_b_id": self.view_b_id,
            "num_matches": self.correspondence.num_matches,
            "num_inliers": self.correspondence.num_inliers,
            "inlier_ratio": round(self.correspondence.inlier_ratio, 3),
            "epipolar_error_pixels": round(self.epipolar_error_pixels, 3),
            "overlap_score": round(self.overlap_score, 3),
            "geometric_consistency_score": round(self.geometric_consistency_score, 3),
            "is_valid_edge": self.is_valid_edge,
        }


@dataclass
class ViewGraph:
    """Directed/undirected connectivity graph of multi-view observations."""
    views: Dict[str, Observation] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], ViewPair] = field(default_factory=dict)
    quality_scores: Dict[str, ViewQualityScore] = field(default_factory=dict)

    def add_view(self, observation: Observation, quality: Optional[ViewQualityScore] = None) -> None:
        self.views[observation.artifact_id] = observation
        if quality:
            self.quality_scores[observation.artifact_id] = quality

    def add_edge(self, pair: ViewPair) -> None:
        key = (pair.view_a_id, pair.view_b_id)
        self.edges[key] = pair

    def get_neighbors(self, view_id: str) -> List[str]:
        neighbors = []
        for (va, vb) in self.edges.keys():
            if va == view_id:
                neighbors.append(vb)
            elif vb == view_id:
                neighbors.append(va)
        return list(set(neighbors))
