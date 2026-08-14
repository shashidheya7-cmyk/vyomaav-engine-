import unittest
from pathlib import Path
import numpy as np
from vyomaa.multiview.contracts import ViewSet, CameraEstimate
from vyomaa.representations.segmentation import SegmentationSet, SegmentationMask
from vyomaa.representations.fused_world import FusedWorldGeometry
from vyomaa.fusion.dense_point_fusion import DenseWorldFusionEngine
from vyomaa.fusion.depth_cross_check import DepthCrossCheck

class TestPhase4CFusion(unittest.TestCase):
    def test_depth_cross_check(self):
        d1 = np.ones((50, 50), dtype=np.float32) * 2.0
        d2 = np.ones((50, 50), dtype=np.float32) * 4.0
        res = DepthCrossCheck.evaluate(d1, d2)
        self.assertAlmostEqual(res["median_ratio"], 0.5, places=2)
        self.assertLess(res["disagreement_percentage"], 10.0)

    def test_dynamic_object_separation_and_ply_export(self):
        cam = CameraEstimate(
            camera_id="cam_0",
            intrinsics_k=np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1]], dtype=np.float32),
            extrinsics_rt=np.eye(4, dtype=np.float32)[:3, :],
            focal_lengths=(500.0, 500.0),
            principal_point=(256.0, 256.0),
            backend_name="VGGT"
        )
        depth = np.ones((512, 512), dtype=np.float32) * 3.0
        mask_arr = np.zeros((512, 512), dtype=bool)
        mask_arr[100:200, 100:200] = True
        seg_mask = SegmentationMask(mask_id="m1", object_id="dynamic_car", mask_array=mask_arr, confidence=0.95)
        seg_set = SegmentationSet(observation_id="obs_0", masks=[seg_mask], tracked_ids=["dynamic_car"])

        engine = DenseWorldFusionEngine(voxel_size=0.05)
        vs = ViewSet(observation_ids=["obs_0"], timestamps=[0.0], keyframe_flags=[True], image_paths=[])
        fused = engine.fuse_multiview(
            view_set=vs,
            cameras=[cam],
            vggt_depths=[depth],
            sam2_segmentations=[seg_set]
        )

        self.assertIsInstance(fused, FusedWorldGeometry)
        self.assertGreater(len(fused.fused_points), 0)
        self.assertIn("dynamic_car", fused.dynamic_clusters)
        self.assertGreater(len(fused.dynamic_clusters["dynamic_car"].points), 0)

        ply_path = "outputs/fused_world/test_output.ply"
        out_file = fused.export_ply(ply_path)
        self.assertTrue(Path(out_file).exists())

if __name__ == "__main__":
    unittest.main()
