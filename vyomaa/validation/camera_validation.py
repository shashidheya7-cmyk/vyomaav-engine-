import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger("vyomaa.validation.camera_validation")

class CameraValidator:
    @staticmethod
    def validate_camera_matrix(k: np.ndarray, rt: np.ndarray, image_size: tuple[int, int]) -> Dict[str, Any]:
        errors = []

        if not np.isfinite(k).all() or not np.isfinite(rt).all():
            raise ValueError("Camera intrinsics or extrinsics contain NaN or Inf values.")

        if k.shape != (3, 3):
            errors.append(f"Invalid intrinsics shape {k.shape}, expected (3, 3).")

        fx, fy = k[0, 0], k[1, 1]
        if fx <= 0 or fy <= 0:
            errors.append(f"Non-positive focal lengths detected: fx={fx}, fy={fy}.")

        cx, cy = k[0, 2], k[1, 2]
        width, height = image_size
        if not (0 <= cx <= width and 0 <= cy <= height):
            errors.append(f"Principal point ({cx}, {cy}) outside image dimensions ({width}x{height}).")

        r = rt[:3, :3]
        det = np.linalg.det(r)
        if abs(det - 1.0) > 1e-2:
            errors.append(f"Rotation matrix determinant is {det:.4f}, expected approx +1.0.")

        ortho_error = np.linalg.norm(r @ r.T - np.eye(3))
        if ortho_error > 1e-2:
            errors.append(f"Rotation matrix is not orthogonal (ortho error: {ortho_error:.4f}).")

        if errors:
            return {"valid": False, "errors": errors}

        return {"valid": True, "errors": []}
