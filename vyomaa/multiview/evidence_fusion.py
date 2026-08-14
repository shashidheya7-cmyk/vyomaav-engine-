"""Multi-view geometric evidence aggregation from genuine observations."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

from ..core.contracts import Camera, Observation
from ..representations.point_cloud import PointCloud
from .view_graph import ViewGraph


class MultiViewEvidenceFusion:
    """Triangulates consistent 3D point tracks across validated view pairs."""

    @staticmethod
    def triangulate_view_pair(
        obs_a: Observation,
        obs_b: Observation,
        points_2d_a: np.ndarray,
        points_2d_b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linear DLT triangulation for paired 2D coordinates."""
        if len(points_2d_a) == 0 or obs_a.camera is None or obs_b.camera is None:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)

        # Projection matrices P = K [R | t]
        P1 = obs_a.camera.K @ obs_a.camera.RT[:3]
        P2 = obs_b.camera.K @ obs_b.camera.RT[:3]

        pts4d = []
        for p1, p2 in zip(points_2d_a, points_2d_b):
            A = np.array([
                p1[0] * P1[2] - P1[0],
                p1[1] * P1[2] - P1[1],
                p2[0] * P2[2] - P2[0],
                p2[1] * P2[2] - P2[1],
            ], dtype=np.float32)

            _, _, Vh = np.linalg.svd(A)
            X = Vh[-1]
            X /= (X[3] if abs(X[3]) > 1e-8 else 1.0)
            pts4d.append(X[:3])

        points_3d = np.array(pts4d, dtype=np.float32)

        # Check positive depth constraint
        valid_depth = points_3d[:, 2] > 0
        return points_3d, valid_depth
