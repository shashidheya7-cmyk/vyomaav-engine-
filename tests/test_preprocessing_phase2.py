"""Unit tests for Phase 2 image preprocessing."""

import unittest
import numpy as np
from PIL import Image

from vyomaa.core.contracts import Camera
from vyomaa.preprocessing.aspect_ratio import AspectRatioNormalizer
from vyomaa.preprocessing.color_normalizer import ColorNormalizer
from vyomaa.preprocessing.lens_undistortion import LensUndistortion


class TestPreprocessingPhase2(unittest.TestCase):

    def test_color_normalization_roundtrip(self):
        srgb_orig = np.array([[[128, 64, 200]]], dtype=np.uint8)
        linear = ColorNormalizer.srgb_to_linear(srgb_orig)
        srgb_rec = ColorNormalizer.linear_to_srgb(linear)
        np.testing.assert_allclose(srgb_orig, srgb_rec, atol=1)

    def test_aspect_ratio_padding(self):
        img = Image.new("RGB", (400, 200), color=(255, 0, 0))
        padded, meta = AspectRatioNormalizer.pad_to_target(img, target_size=(512, 512))
        self.assertEqual(padded.size, (512, 512))
        self.assertEqual(meta.original_width, 400)
        self.assertEqual(meta.original_height, 200)
        self.assertGreaterEqual(meta.pad_left, 0)
        self.assertGreaterEqual(meta.pad_top, 0)

    def test_lens_undistortion(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cam = Camera(image_width=640, image_height=480)
        cam.distortion_coefficients = [-0.1, 0.01, 0.0, 0.0, 0.0]

        undistorted, new_cam = LensUndistortion.undistort_image(img, cam)
        self.assertEqual(undistorted.shape, (480, 640, 3))
        self.assertEqual(new_cam.distortion_coefficients, [0.0, 0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
