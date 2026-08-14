"""Epipolar geometry validation and relative pose recovery via Fundamental / Essential matrices."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from ..core.contracts import Camera, Observation
from ..core.exceptions import MultiViewError
from .view_graph import CorrespondenceMap, ViewPair


class EpipolarChecker:
    """Validates epipolar consistency using RANSAC and computes relative rotation & translation."""

    @staticmethod
    def validate_pair(
        obs_a: Observation,
        obs_b: Observation,
        correspondence: CorrespondenceMap,
        ransac_thresh_pixels: float = 3.0,
    ) -> ViewPair:
        """Estimate Essential Matrix, recover pose, and filter non-epipolar outliers."""
        pts_a, pts_b = correspondence.points_a, correspondence.points_b

        if len(pts_a) < 8 or obs_a.camera is None or obs_b.camera is None:
            # Insufficient correspondences
            return ViewPair(
                view_a_id=obs_a.artifact_id,
                view_b_id=obs_b.artifact_id,
                correspondence=correspondence,
                relative_R=np.eye(3, dtype=np.float32),
                relative_t=np.zeros(3, dtype=np.float32),
                epipolar_error_pixels=999.0,
                overlap_score=0.0,
                geometric_consistency_score=0.0,
                is_valid_edge=False,
            )

        K_a, K_b = obs_a.camera.K, obs_b.camera.K

        # Normalize coordinates
        E, inlier_mask = cv2.findEssentialMat(
            pts_a, pts_b, K_a, method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh_pixels
        )

        if E is None or inlier_mask is None:
            correspondence.inlier_mask = np.zeros(len(pts_a), dtype=bool)
            return ViewPair(
                view_a_id=obs_a.artifact_id,
                view_b_id=obs_b.artifact_id,
                correspondence=correspondence,
                relative_R=np.eye(3, dtype=np.float32),
                relative_t=np.zeros(3, dtype=np.float32),
                epipolar_error_pixels=999.0,
                overlap_score=0.0,
                geometric_consistency_score=0.0,
                is_valid_edge=False,
            )

        mask = (inlier_mask.ravel() == 1)
        correspondence.inlier_mask = mask

        inlier_pts_a = pts_a[mask]
        inlier_pts_b = pts_b[mask]

        if len(inlier_pts_a) < 5:
            return ViewPair(
                view_a_id=obs_a.artifact_id,
                view_b_id=obs_b.artifact_id,
                correspondence=correspondence,
                relative_R=np.eye(3, dtype=np.float32),
                relative_t=np.zeros(3, dtype=np.float32),
                epipolar_error_pixels=999.0,
                overlap_score=0.0,
                geometric_consistency_score=0.0,
                is_valid_edge=False,
            )

        # Recover relative pose
        _, R_rel, t_rel, pose_mask = cv2.recoverPose(E, inlier_pts_a, inlier_pts_b, K_a)

        # Sampson epipolar distance computation
        F, _ = cv2.findFundamentalMat(inlier_pts_a, inlier_pts_b, method=cv2.FM_8POINT)
        epipolar_errors = []
        if F is not None:
            for pa, pb in zip(inlier_pts_a, inlier_pts_b):
                pt1 = np.array([pa[0], pa[1], 1.0])
                pt2 = np.array([pb[0], pb[1], 1.0])
                line2 = F @ pt1
                denom = line2[0]**2 + line2[1]**2
                if denom > 1e-8:
                    dist = abs(np.dot(pt2, line2)) / np.sqrt(denom)
                    epipolar_errors.append(dist)

        mean_err = float(np.mean(epipolar_errors)) if epipolar_errors else 1.0
        overlap = min(1.0, float(len(inlier_pts_a)) / 300.0)
        geom_score = max(0.0, min(1.0, (1.0 - (mean_err / ransac_thresh_pixels)) * correspondence.inlier_ratio))

        return ViewPair(
            view_a_id=obs_a.artifact_id,
            view_b_id=obs_b.artifact_id,
            correspondence=correspondence,
            relative_R=R_rel.astype(np.float32),
            relative_t=t_rel.ravel().astype(np.float32),
            epipolar_error_pixels=round(mean_err, 3),
            overlap_score=round(overlap, 3),
            geometric_consistency_score=round(geom_score, 3),
            is_valid_edge=(len(inlier_pts_a) >= 12 and mean_err < ransac_thresh_pixels * 1.5),
        )
