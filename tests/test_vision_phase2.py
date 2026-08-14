import unittest

from PIL import Image
import numpy as np

from vyomaa.core.contracts import FrameArtifact
from vyomaa.core.exceptions import ModelUnavailableError
from vyomaa.vision.bytetrack_tracker import ByteTrackTracker
from vyomaa.vision.depth_anything import DepthAnythingV2Adapter
from vyomaa.vision.sam2_segmentation import SAM2SegmentationAdapter
from vyomaa.vision.surface_normals import SurfaceNormalEstimator


class TestVisionPhase2(unittest.TestCase):

    def test_bytetrack_association(self):
        tracker = ByteTrackTracker()
        tracker.initialize()

        frame = FrameArtifact(
            name="test_frame_000000",
            frame_index=0,
            timestamp_seconds=0.0,
            sequence_id="test_sequence",
            image_path=None,
            resolution=(64, 64),
        )

        detections = [
            ([10, 10, 50, 50], 0.9),
        ]

        tracks = tracker.track_frame(frame, detections)

        self.assertIsInstance(tracks, list)
        self.assertEqual(len(tracks), 1)

        frame2 = FrameArtifact(
            name="test_frame_000001",
            frame_index=1,
            timestamp_seconds=1.0 / 30.0,
            sequence_id="test_sequence",
            image_path=None,
            resolution=(64, 64),
        )

        tracks2 = tracker.track_frame(
            frame2,
            [([11, 11, 51, 51], 0.9)],
        )

        self.assertIsInstance(tracks2, list)
        self.assertEqual(len(tracks2), 1)

        # Same physical detection should retain the same identity.
        self.assertEqual(
            tracks2[0].track_id,
            tracks[0].track_id,
        )

    def test_depth_anything_uninitialized_raises(self):
        adapter = DepthAnythingV2Adapter()

        image = Image.fromarray(
            np.zeros((64, 64, 3), dtype=np.uint8)
        )

        with self.assertRaises(ModelUnavailableError):
            adapter.estimate_depth(image)

    def test_sam2_uninitialized_raises(self):
        adapter = SAM2SegmentationAdapter()

        image = Image.fromarray(
            np.zeros((64, 64, 3), dtype=np.uint8)
        )

        with self.assertRaises(ModelUnavailableError):
            adapter.segment_image(image)

    def test_surface_normal_estimation(self):
        depth = np.ones((32, 32), dtype=np.float32)

        normals, confidence = (
            SurfaceNormalEstimator.compute_normals_from_depth(
                depth,
                None,
            )
        )

        self.assertEqual(normals.shape, (32, 32, 3))
        self.assertEqual(confidence.shape, (32, 32))


if __name__ == "__main__":
    unittest.main()
