"""Unit tests for Phase 2 camera geometry, bundle adjustment, and scale estimation."""

import unittest
import numpy as np

from vyomaa.camera_geometry.bundle_adjustment import BundleAdjuster
from vyomaa.camera_geometry.geometry_router import GeometryRouter
from vyomaa.camera_geometry.scale_estimator import ScaleEstimator, ScaleSource, ScaleConfidence
from vyomaa.core.contracts import Camera, InputArtifact
from vyomaa.core.types import ModalityType


class TestCameraGeometryPhase2(unittest.TestCase):

    def test_scale_estimator(self):
        # Metric sensor
        scale_metric = ScaleEstimator.estimate_scale(is_metric_sensor=True)
        self.assertEqual(scale_metric.source, ScaleSource.SENSOR_METRIC_RGBD)
        self.assertEqual(scale_metric.confidence, ScaleConfidence.EXACT)

        # Semantic prior
        scale_cup = ScaleEstimator.estimate_scale(is_metric_sensor=False, semantic_class="cup", observed_bounds_extent=1.0)
        self.assertEqual(scale_cup.source, ScaleSource.ESTIMATED_OBJECT_PRIOR)
        self.assertAlmostEqual(scale_cup.scale_factor_to_meters, 0.12, places=3)

    def test_geometry_router(self):
        art_single = InputArtifact(modality=ModalityType.RGB_IMAGE)
        self.assertEqual(GeometryRouter.select_solver(art_single), "single_image_perspective")

        art_mv = InputArtifact(modality=ModalityType.MULTIVIEW_IMAGE_SET)
        self.assertEqual(GeometryRouter.select_solver(art_mv), "multiview_correspondence_sfm")

        art_vid = InputArtifact(modality=ModalityType.MONOCULAR_VIDEO)
        self.assertEqual(GeometryRouter.select_solver(art_vid), "video_temporal_sfm")

    def test_bundle_adjuster_convergence(self):
        # 2 cameras observing 4 3D points
        cam1 = Camera(name="Cam1", image_width=640, image_height=480)
        cam2 = Camera(name="Cam2", image_width=640, image_height=480)
        # Cam2 slightly translated
        cam2.extrinsic_matrix[0][3] = 0.5

        pts_3d = np.array([
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [0.0, 0.5, 2.0],
            [0.0, 0.0, 2.5],
        ], dtype=np.float32)

        # Compute synthetic ground truth 2D projections
        cam_indices = []
        point_indices = []
        points_2d = []

        for c_i, cam in enumerate([cam1, cam2]):
            for p_i, pt in enumerate(pts_3d):
                p_cam = cam.RT[:3, :3] @ pt + cam.RT[:3, 3]
                u = cam.focal_length_x * (p_cam[0] / p_cam[2]) + cam.principal_point_x
                v = cam.focal_length_y * (p_cam[1] / p_cam[2]) + cam.principal_point_y
                cam_indices.append(c_i)
                point_indices.append(p_i)
                points_2d.append([u, v])

        cam_idx_arr = np.array(cam_indices, dtype=np.int32)
        pt_idx_arr = np.array(point_indices, dtype=np.int32)
        pts_2d_arr = np.array(points_2d, dtype=np.float32)

        # Add slight initial perturbation to 3D points
        noisy_pts_3d = pts_3d + np.random.normal(0, 0.05, pts_3d.shape).astype(np.float32)

        res = BundleAdjuster.optimize(
            cameras=[cam1, cam2],
            points_3d=noisy_pts_3d,
            camera_indices=cam_idx_arr,
            point_indices=pt_idx_arr,
            points_2d=pts_2d_arr,
            max_nfev=30,
        )

        self.assertTrue(res.is_converged)
        self.assertLess(res.final_reprojection_error_pixels, 1.0)


if __name__ == "__main__":
    unittest.main()
