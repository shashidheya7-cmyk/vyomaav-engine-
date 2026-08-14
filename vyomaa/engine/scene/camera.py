
"""Camera state representation."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Camera:
    """Pinhole camera represented by float32 intrinsic and extrinsic matrices."""

    intrinsic_matrix: np.ndarray
    extrinsic_matrix: np.ndarray

    def __post_init__(self) -> None:
        self.intrinsic_matrix = np.ascontiguousarray(self.intrinsic_matrix, dtype=np.float32)
        self.extrinsic_matrix = np.ascontiguousarray(self.extrinsic_matrix, dtype=np.float32)
        if self.intrinsic_matrix.shape != (3, 3):
            raise ValueError("intrinsic_matrix must have shape (3, 3)")
        if self.extrinsic_matrix.shape != (4, 4):
            raise ValueError("extrinsic_matrix must have shape (4, 4)")

    @classmethod
    def create_default(cls) -> "Camera":
        """Create an identity camera suitable for synthetic canonical views."""
        return cls(np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32))

    @classmethod
    def look_at(cls, position: np.ndarray, target: np.ndarray, focal_length: float,
                image_size: int) -> "Camera":
        """Build a right-handed Y-up world-to-camera pinhole camera."""
        eye = np.asarray(position, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        rotation = np.stack((right, up, forward))
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3], extrinsic[:3, 3] = rotation, -rotation @ eye
        principal = (image_size - 1) / 2.0
        intrinsic = np.array([[focal_length, 0, principal], [0, focal_length, principal], [0, 0, 1]], dtype=np.float32)
        return cls(intrinsic, extrinsic)


