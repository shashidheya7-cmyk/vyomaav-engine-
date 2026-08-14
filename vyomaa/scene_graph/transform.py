"""3D affine transformation primitives supporting rigid composition and inversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np

from ..core.exceptions import SceneGraphError


@dataclass
class Transform3D:
    """3D Rigid/Affine transformation parameterized by Translation, Rotation, and Scale."""

    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_quaternion: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    def to_matrix(self) -> np.ndarray:
        """Compute the 4x4 homogeneous transformation matrix."""
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = self.translation

        # Quaternion to rotation matrix
        x, y, z, w = self.rotation_quaternion
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm < 1e-8:
            R = np.eye(3, dtype=np.float32)
        else:
            x, y, z, w = x / norm, y / norm, z / norm, w / norm
            R = np.array([
                [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
                [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
                [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)],
            ], dtype=np.float32)

        # Scale
        S = np.diag(self.scale)
        RS = R @ S

        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = RS
        M[:3, 3] = self.translation
        return M

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Transform3D:
        """Decompose a 4x4 affine matrix into translation, quaternion, and scale."""
        if matrix.shape != (4, 4):
            raise SceneGraphError(f"Matrix must be (4, 4), got {matrix.shape}")

        translation = matrix[:3, 3].tolist()
        RS = matrix[:3, :3]

        # Extract scale
        sx = float(np.linalg.norm(RS[:, 0]))
        sy = float(np.linalg.norm(RS[:, 1]))
        sz = float(np.linalg.norm(RS[:, 2]))
        scale = [sx, sy, sz]

        R = np.zeros((3, 3), dtype=np.float32)
        R[:, 0] = RS[:, 0] / (sx if sx > 1e-8 else 1.0)
        R[:, 1] = RS[:, 1] / (sy if sy > 1e-8 else 1.0)
        R[:, 2] = RS[:, 2] / (sz if sz > 1e-8 else 1.0)

        # Convert R to quaternion
        trace = np.trace(R)
        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return cls(
            translation=translation,
            rotation_quaternion=[float(x), float(y), float(z), float(w)],
            scale=scale,
        )

    def compose(self, other: Transform3D) -> Transform3D:
        """Compose self * other (applying other first, then self)."""
        M_self = self.to_matrix()
        M_other = other.to_matrix()
        return Transform3D.from_matrix(M_self @ M_other)

    def inverse(self) -> Transform3D:
        """Compute the mathematical inverse transform."""
        M = self.to_matrix()
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError as exc:
            raise SceneGraphError("Singular transform matrix cannot be inverted") from exc
        return Transform3D.from_matrix(M_inv)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Apply transform to an (N, 3) array of 3D points."""
        if len(points) == 0:
            return points
        pts_h = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
        transformed = (self.to_matrix() @ pts_h.T).T
        return transformed[:, :3].astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "translation": self.translation,
            "rotation_quaternion": self.rotation_quaternion,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Transform3D:
        return cls(**data)
