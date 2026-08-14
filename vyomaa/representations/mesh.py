"""Canonical triangle mesh representation with strict array invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np

from ..core.base_artifact import BaseArtifact
from ..core.exceptions import GeometryError, SchemaValidationError
from ..core.metadata import ArtifactMetadata
from ..core.provenance import ProvenanceRecord
from ..core.types import ArtifactType


@dataclass
class MeshData(BaseArtifact):
    """Production triangle mesh container stored in contiguous NumPy arrays."""

    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    faces: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))
    vertex_colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    uvs: Optional[np.ndarray] = None
    uv_faces: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.MESH
        self.set_geometry_data(self.vertices, self.faces, self.vertex_colors, self.normals)
        if self.uvs is not None:
            self.set_uv_data(self.uvs, self.uv_faces)
        super().__post_init__()

    def set_geometry_data(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
    ) -> None:
        """Validate and set vertices, faces, colors, and surface normals."""
        v = np.ascontiguousarray(vertices, dtype=np.float32)
        f = np.ascontiguousarray(faces, dtype=np.int32)
        if v.ndim != 2 or v.shape[1:] != (3,):
            raise SchemaValidationError(f"vertices must have shape (V, 3), got {v.shape}")
        if f.ndim != 2 or f.shape[1:] != (3,):
            raise SchemaValidationError(f"faces must have shape (F, 3), got {f.shape}")
        if len(f) > 0 and (f.min() < 0 or f.max() >= len(v)):
            raise SchemaValidationError(f"Face index out of bounds: range [{f.min()}, {f.max()}] vs V={len(v)}")

        c = None
        if vertex_colors is not None:
            c = np.ascontiguousarray(vertex_colors, dtype=np.float32)
            if c.shape != v.shape:
                raise SchemaValidationError(f"vertex_colors shape {c.shape} must match vertices {v.shape}")

        n = None
        if normals is not None:
            n = np.ascontiguousarray(normals, dtype=np.float32)
            if n.shape != v.shape:
                raise SchemaValidationError(f"normals shape {n.shape} must match vertices {v.shape}")

        self.vertices, self.faces, self.vertex_colors, self.normals = v, f, c, n

    def set_uv_data(self, uvs: np.ndarray, uv_faces: Optional[np.ndarray] = None) -> None:
        """Validate and bind chart-packed UV texture coordinates."""
        u = np.ascontiguousarray(uvs, dtype=np.float32)
        if u.ndim != 2 or u.shape[1:] != (2,):
            raise SchemaValidationError(f"uvs must have shape (U, 2), got {u.shape}")
        uf = np.ascontiguousarray(self.faces if uv_faces is None else uv_faces, dtype=np.int32)
        if uf.shape != self.faces.shape or (len(uf) > 0 and (uf.min() < 0 or uf.max() >= len(u))):
            raise SchemaValidationError(f"uv_faces indexing error against UV coordinate count {len(u)}")
        self.uvs, self.uv_faces = u, uf

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def compute_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute axis-aligned min and max bounding extents."""
        if len(self.vertices) == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def compute_face_normals(self) -> np.ndarray:
        """Calculate normalized face normal vectors."""
        if len(self.faces) == 0 or len(self.vertices) == 0:
            return np.empty((0, 3), dtype=np.float32)
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(cross, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return (cross / norm).astype(np.float32)

    def normalize_to_unit_cube(self) -> MeshData:
        """Center and scale vertices into canonical [-1, 1]^3 domain."""
        if len(self.vertices) == 0:
            return self
        min_pt, max_pt = self.compute_bounds()
        center = (min_pt + max_pt) / 2.0
        extent = float(np.abs(self.vertices - center).max())
        scale = 1.0 if extent < 1e-8 else (1.0 / extent)
        self.vertices = ((self.vertices - center) * scale).astype(np.float32)
        return self

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "has_normals": self.normals is not None,
            "has_colors": self.vertex_colors is not None,
            "has_uvs": self.uvs is not None,
            "bounds_min": self.compute_bounds()[0].tolist(),
            "bounds_max": self.compute_bounds()[1].tolist(),
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MeshData:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.MESH
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        # Clean non-field stats
        for k in ["vertex_count", "face_count", "has_normals", "has_colors", "has_uvs", "bounds_min", "bounds_max"]:
            kwargs.pop(k, None)
        return cls(**kwargs)
