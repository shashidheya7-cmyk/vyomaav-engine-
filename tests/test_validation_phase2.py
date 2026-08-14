"""Unit tests for Phase 2 geometric validation and quality scoring."""

import unittest
import numpy as np

from vyomaa.core.contracts import Camera, DepthMap
from vyomaa.representations.point_cloud import PointCloud
from vyomaa.validation.geometric_validator import GeometricValidator


class TestValidationPhase2(unittest.TestCase):

    def test_depth_map_validation(self):
        depth_art = DepthMap(width=640, height=480, min_depth=0.5, max_depth=5.0)
        valid, score, warnings = GeometricValidator.validate_depth_map(depth_art)
        self.assertTrue(valid)
        self.assertGreater(score, 0.5)

    def test_camera_validation(self):
        cam = Camera(image_width=1920, image_height=1080)
        valid, score, warnings = GeometricValidator.validate_camera(cam)
        self.assertTrue(valid)
        self.assertEqual(score, 1.0)

    def test_point_cloud_validation(self):
        pts = np.random.uniform(-1, 1, (500, 3)).astype(np.float32)
        pc = PointCloud(points=pts)
        valid, score, warnings = GeometricValidator.validate_point_cloud(pc)
        self.assertTrue(valid)
        self.assertGreater(score, 0.4)

    def test_comprehensive_validation_report(self):
        depth_art = DepthMap(width=640, height=480, min_depth=0.5, max_depth=5.0)
        cam = Camera(image_width=640, image_height=480)
        pc = PointCloud(points=np.random.uniform(-1, 1, (1000, 3)).astype(np.float32))

        report = GeometricValidator.generate_comprehensive_report(
            depth_maps=[depth_art],
            cameras=[cam],
            point_cloud=pc,
        )
        self.assertTrue(report.is_valid)
        self.assertGreater(report.overall_quality_score, 0.7)


if __name__ == "__main__":
    unittest.main()
