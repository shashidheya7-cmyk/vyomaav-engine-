"""Unit tests for MeshData validation, face normals, and unit cube normalization."""

import unittest
import numpy as np

from vyomaa.core.exceptions import SchemaValidationError
from vyomaa.representations.mesh import MeshData


class TestMeshData(unittest.TestCase):

    def test_mesh_data_invariants(self):
        # Single tetrahedron
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2],
        ], dtype=np.int32)

        mesh = MeshData(name="Tetrahedron", vertices=vertices, faces=faces)
        self.assertEqual(mesh.vertex_count, 4)
        self.assertEqual(mesh.face_count, 4)

        normals = mesh.compute_face_normals()
        self.assertEqual(normals.shape, (4, 3))

        # Normalize
        mesh.normalize_to_unit_cube()
        min_b, max_b = mesh.compute_bounds()
        self.assertTrue(np.all(min_b >= -1.0001))
        self.assertTrue(np.all(max_b <= 1.0001))

    def test_invalid_face_indices(self):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        faces = np.array([[0, 1, 5]], dtype=np.int32)  # Index 5 out of bounds
        with self.assertRaises(SchemaValidationError):
            MeshData(vertices=vertices, faces=faces)


if __name__ == "__main__":
    unittest.main()
