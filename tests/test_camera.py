"""Unit tests for Camera contracts and matrix representations."""

import unittest
import numpy as np

from vyomaa.core.contracts import Camera


class TestCamera(unittest.TestCase):

    def test_camera_matrices_and_serialization(self):
        K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        RT = np.eye(4, dtype=np.float32)

        cam = Camera.from_matrices(K, RT, image_size=(640, 480), name="Front Cam")
        self.assertEqual(cam.image_width, 640)
        self.assertEqual(cam.image_height, 480)
        np.testing.assert_allclose(cam.K, K)
        np.testing.assert_allclose(cam.RT, RT)

        d = cam.to_dict()
        cam_re = Camera.from_dict(d)
        np.testing.assert_allclose(cam_re.K, K)


if __name__ == "__main__":
    unittest.main()
