"""Unit tests for Phase 2 neural vision adapters and surface normal estimation."""

import unittest
import numpy as np

from vyomaa.core.contracts import Camera, FrameArtifact
from vyomaa.core.exceptions import ModelUnavailableError
from vyomaa.vision.bytetrack_tracker import ByteTrackTracker
from vyomaa.vision.depth_anything import DepthAnythingV2Adapter
from vyomaa.vision.sam2_segmentation import SAM2SegmentationAdapter
from vyomaa.vision.surface_normals import SurfaceNormalEstimator


class TestVisionPhase2(unittest.TestCase):

    def test_depth_anything_uninitialized_raises(self):
        adapter = DepthAnythingV2Adapter()
        with self.assertRaises(ModelUnavailableError):
            adapter.initialize(device="cuda")

    def test_sam2_uninitialized_raises(self):
        adapter = SAM2SegmentationAdapter()
        with self.assertRaises(ModelUnavailableError):
            adapter.initialize(device="cuda")

    def test_bytetrack_association(self):
        tracker = ByteTrackTracker()
        tracker.initialize()

        fa1 = FrameArtifact(frame_index=0)
        dets_frame1 = [([10.0, 10.0, 50.0, 50.0], 0.95)]
        tracks1 = tracker.track_frame(fa1, dets_frame1)
        self.assertEqual(len(tracks1), 1)
        t1_id = tracks1[0].track_id

        # Frame 2: Slightly shifted detection
        fa2 = FrameArtifact(frame_index=1)
        dets_frame2 = [([12.0, 12.0, 52.0, 52.0], 0.94)]
        tracks2 = tracker.track_frame(fa2, dets_frame2)
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].track_id, t1_id)  # Persistent track ID

    def test_surface_normal_estimation(self):
        # Flat planar depth map
        depth = np.ones((100, 100), dtype=np.float32) * 2.0
        cam = Camera(image_width=100, image_height=100, focal_length_x=100.0, focal_length_y=100.0)

        normals, conf = SurfaceNormalEstimator.compute_normals_from_depth(depth, cam)
        self.assertEqual(normals.shape, (100, 100, 3))
        # Normals for flat surface should point along +Z [0, 0, 1]
        np.testing.assert_allclose(normals[50, 50], [0.0, 0.0, 1.0], atol=1e-3)
        self.assertTrue(np.all(conf >= 0.0))


if __name__ == "__main__":
    unittest.main()
