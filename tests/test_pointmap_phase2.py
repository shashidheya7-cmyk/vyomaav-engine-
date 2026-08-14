"""Unit tests for Phase 2 3D back-projection and point cloud merging."""

import unittest
import numpy as np

from vyomaa.core.contracts import Camera, Observation
from vyomaa.pointmap.backprojector import DepthBackprojector


class TestPointMapPhase2(unittest.TestCase):

    def test_backprojection_exact_coordinates(self):
        # Camera at origin with focal=100, image 100x100, principal=(50, 50)
        K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        RT = np.eye(4, dtype=np.float32)
        cam = Camera.from_matrices(K, RT, (100, 100))

        obs = Observation(name="TestObs", camera=cam)

        # Depth of 2.0 meters everywhere
        depth = np.ones((100, 100), dtype=np.float32) * 2.0

        pc = DepthBackprojector.backproject_observation(obs, depth, stride=10)
        self.assertGreater(pc.point_count, 0)

        # Center point at (50, 50) should unproject to (0, 0, 2.0)
        # Find point closest to center
        min_dist = float(np.min(np.linalg.norm(pc.points - np.array([0.0, 0.0, 2.0]), axis=1)))
        self.assertAlmostEqual(min_dist, 0.0, delta=0.1)

    def test_point_cloud_merging(self):
        # 2 non-empty point clouds
        cam = Camera(image_width=50, image_height=50)
        obs1 = Observation(name="Obs1", camera=cam)
        obs2 = Observation(name="Obs2", camera=cam)

        d1 = np.ones((50, 50), dtype=np.float32) * 1.5
        d2 = np.ones((50, 50), dtype=np.float32) * 2.5

        pc1 = DepthBackprojector.backproject_observation(obs1, d1, stride=5)
        pc2 = DepthBackprojector.backproject_observation(obs2, d2, stride=5)

        merged = DepthBackprojector.merge_point_clouds([pc1, pc2], voxel_size=0.01)
        self.assertGreater(merged.point_count, pc1.point_count)


if __name__ == "__main__":
    unittest.main()
