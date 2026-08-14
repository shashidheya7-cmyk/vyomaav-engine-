"""Visual feature correspondence extraction using OpenCV SIFT and ORB."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..core.contracts import Observation
from ..core.exceptions import MultiViewError
from .view_graph import CorrespondenceMap, ViewQualityScore


class CorrespondenceEngine:
    """Extracts invariant 2D visual descriptors and matches them with Lowe's ratio test."""

    def __init__(self, detector_type: str = "SIFT", max_features: int = 4000) -> None:
        self.detector_type = detector_type.upper()
        self.max_features = max_features

        if self.detector_type == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def extract_features(self, image_bgr: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect keypoints and compute local feature descriptors."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
        kps, descs = self.detector.detectAndCompute(gray, None)
        return kps, descs if descs is not None else np.empty((0, 128), dtype=np.float32)

    def evaluate_quality(self, view_id: str, image_bgr: np.ndarray) -> ViewQualityScore:
        """Compute Laplacian sharpness and feature density metrics."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        kps, _ = self.extract_features(image_bgr)
        feature_density = len(kps)
        mean_lum = float(np.mean(gray))
        exposure_balance = 1.0 - abs(mean_lum - 128.0) / 128.0

        overall = min(1.0, (sharpness / 500.0) * 0.4 + (feature_density / 2000.0) * 0.4 + exposure_balance * 0.2)
        return ViewQualityScore(
            view_id=view_id,
            sharpness=round(sharpness, 2),
            exposure_balance=round(exposure_balance, 3),
            feature_density=feature_density,
            overall_quality=round(overall, 3),
        )

    def match_views(
        self,
        obs_a: Observation,
        img_a: np.ndarray,
        obs_b: Observation,
        img_b: np.ndarray,
        ratio_thresh: float = 0.75,
    ) -> CorrespondenceMap:
        """Match two views and apply ratio test outlier rejection."""
        kps_a, descs_a = self.extract_features(img_a)
        kps_b, descs_b = self.extract_features(img_b)

        if len(descs_a) == 0 or len(descs_b) == 0:
            return CorrespondenceMap(
                view_a_id=obs_a.artifact_id,
                view_b_id=obs_b.artifact_id,
                points_a=np.empty((0, 2), dtype=np.float32),
                points_b=np.empty((0, 2), dtype=np.float32),
                match_scores=np.empty(0, dtype=np.float32),
                inlier_mask=np.empty(0, dtype=bool),
            )

        knn_matches = self.matcher.knnMatch(descs_a, descs_b, k=2)
        good_pts_a, good_pts_b, scores = [], [], []

        for m_pair in knn_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < ratio_thresh * n.distance:
                    good_pts_a.append(kps_a[m.queryIdx].pt)
                    good_pts_b.append(kps_b[m.trainIdx].pt)
                    scores.append(1.0 - (m.distance / (n.distance + 1e-6)))

        pts_a = np.array(good_pts_a, dtype=np.float32) if good_pts_a else np.empty((0, 2), dtype=np.float32)
        pts_b = np.array(good_pts_b, dtype=np.float32) if good_pts_b else np.empty((0, 2), dtype=np.float32)
        match_scores = np.array(scores, dtype=np.float32) if scores else np.empty(0, dtype=np.float32)
        inlier_mask = np.ones(len(pts_a), dtype=bool)

        return CorrespondenceMap(
            view_a_id=obs_a.artifact_id,
            view_b_id=obs_b.artifact_id,
            points_a=pts_a,
            points_b=pts_b,
            match_scores=match_scores,
            inlier_mask=inlier_mask,
        )
