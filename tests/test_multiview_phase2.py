import cv2
"""Unit tests for Phase 2 multi-view correspondence, epipolar geometry, and view selection."""

import unittest
import numpy as np

from vyomaa.core.contracts import Camera, Observation
from vyomaa.multiview.correspondence import CorrespondenceEngine
from vyomaa.multiview.epipolar_checker import EpipolarChecker
from vyomaa.multiview.view_graph import ViewGraph, ViewPair, CorrespondenceMap
from vyomaa.multiview.view_selector import ViewSelector


class TestMultiViewPhase2(unittest.TestCase):

    def test_sift_feature_extraction_and_matching(self):
        engine = CorrespondenceEngine(detector_type="SIFT", max_features=1000)

        # Create two textured patterns with known translation
        img_a = np.zeros((300, 300, 3), dtype=np.uint8)
        for i in range(10, 280, 20):
            cv2.rectangle(img_a, (i, i), (i + 15, i + 15), (255, 255, 255), -1)

        # Shifted by (10, 5)
        M = np.float32([[1, 0, 10], [0, 1, 5]])
        img_b = cv2.warpAffine(img_a, M, (300, 300))

        obs_a = Observation(name="ViewA", frame_id="view_a")
        obs_b = Observation(name="ViewB", frame_id="view_b")

        corr = engine.match_views(obs_a, img_a, obs_b, img_b)
        self.assertGreater(corr.num_matches, 10)

    def test_view_selector(self):
        vg = ViewGraph()
        obs1 = Observation(name="O1")
        obs2 = Observation(name="O2")
        vg.add_view(obs1)
        vg.add_view(obs2)

        # High quality edge
        corr = CorrespondenceMap(
            view_a_id=obs1.artifact_id,
            view_b_id=obs2.artifact_id,
            points_a=np.ones((20, 2), dtype=np.float32),
            points_b=np.ones((20, 2), dtype=np.float32),
            match_scores=np.ones(20, dtype=np.float32),
            inlier_mask=np.ones(20, dtype=bool),
        )
        pair = ViewPair(
            view_a_id=obs1.artifact_id,
            view_b_id=obs2.artifact_id,
            correspondence=corr,
            relative_R=np.eye(3, dtype=np.float32),
            relative_t=np.zeros(3, dtype=np.float32),
            epipolar_error_pixels=0.5,
            overlap_score=0.6,
            geometric_consistency_score=0.9,
            is_valid_edge=True,
        )
        vg.add_edge(pair)

        selected = ViewSelector.select_informative_views(vg, min_overlap=0.2, max_redundancy_overlap=0.9)
        self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    import cv2
    unittest.main()
