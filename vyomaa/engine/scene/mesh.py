
"""Mesh container with strict array invariants."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class MeshData:
    """Triangle mesh stored in contiguous normalized NumPy arrays."""

    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    faces: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))
    vertex_colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    uvs: Optional[np.ndarray] = None
    uv_faces: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.set_data(self.vertices, self.faces, self.vertex_colors, self.normals)
        if self.uvs is not None:
            self.set_uv_data(self.uvs, self.uv_faces)

    def set_data(self, vertices: np.ndarray, faces: np.ndarray,
                 vertex_colors: Optional[np.ndarray] = None,
                 normals: Optional[np.ndarray] = None) -> None:
        """Validate and set mesh data with canonical contiguous dtypes."""
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        faces = np.ascontiguousarray(faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise ValueError("vertices must have shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1:] != (3,):
            raise ValueError("faces must have shape (F, 3)")
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("faces contain indices outside vertices")
        colors = None
        if vertex_colors is not None:
            colors = np.ascontiguousarray(vertex_colors, dtype=np.float32)
            if colors.shape != vertices.shape:
                raise ValueError("vertex_colors must have shape (V, 3)")
        normal_array = None
        if normals is not None:
            normal_array = np.ascontiguousarray(normals, dtype=np.float32)
            if normal_array.shape != vertices.shape:
                raise ValueError("normals must have shape (V, 3)")
        self.vertices, self.faces, self.vertex_colors, self.normals = vertices, faces, colors, normal_array

    def set_uv_data(self, uvs: np.ndarray, uv_faces: Optional[np.ndarray] = None) -> None:
        """Set chart-packed UV coordinates and triangular UV indices."""
        coordinates = np.ascontiguousarray(uvs, dtype=np.float32)
        indices = np.ascontiguousarray(self.faces if uv_faces is None else uv_faces, dtype=np.int32)
        if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
            raise ValueError("uvs must have shape (U, 2)")
        if indices.shape != self.faces.shape or (len(indices) and (indices.min() < 0 or indices.max() >= len(coordinates))):
            raise ValueError("uv_faces must index UV coordinates for every mesh face")
        self.uvs, self.uv_faces = coordinates, indices

    @property
    def vertex_count(self) -> int:
        """Number of vertices."""
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        """Number of triangular faces."""
        return int(self.faces.shape[0])


