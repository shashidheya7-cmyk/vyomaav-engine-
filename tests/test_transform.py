"""Unit tests for 3D Transform composition, inversion, and point transformations."""

import unittest
import numpy as np

from vyomaa.scene_graph.transform import Transform3D


class TestTransform(unittest.TestCase):

    def test_transform_matrix_and_inversion(self):
        t1 = Transform3D(translation=[2.0, -3.0, 5.0], scale=[1.0, 1.0, 1.0])
        t1_inv = t1.inverse()

        comp = t1.compose(t1_inv)
        mat = comp.to_matrix()
        np.testing.assert_allclose(mat, np.eye(4), atol=1e-5)

    def test_point_transformation(self):
        t = Transform3D(translation=[10.0, 0.0, 0.0], scale=[2.0, 2.0, 2.0])
        points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        transformed = t.transform_points(points)

        expected = np.array([[12.0, 4.0, 6.0]], dtype=np.float32)
        np.testing.assert_allclose(transformed, expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
