"""Lens distortion removal and camera calibration adjustment using OpenCV."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..core.contracts import Camera
from ..core.exceptions import PreprocessingError


class LensUndistortion:
    """Removes radial and tangential lens distortion and computes updated camera intrinsics."""

    @staticmethod
    def undistort_image(
        image_bgr: np.ndarray,
        camera: Camera,
        alpha: float = 0.0,  # 0: zoom to valid pixels, 1: keep all pixels
    ) -> Tuple[np.ndarray, Camera]:
        """Undistort image and return corrected image and updated Camera contract."""
        h, w = image_bgr.shape[:2]
        K = camera.K
        dist_coeffs = np.array(camera.distortion_coefficients, dtype=np.float32) if camera.distortion_coefficients else np.zeros(5, dtype=np.float32)

        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha, (w, h))
        undistorted = cv2.undistort(image_bgr, K, dist_coeffs, None, new_K)

        new_camera = Camera.from_matrices(
            K=new_K.astype(np.float32),
            RT=camera.RT,
            image_size=(w, h),
            name=f"{camera.name}_undistorted",
        )
        new_camera.distortion_coefficients = [0.0, 0.0, 0.0, 0.0, 0.0]

        return undistorted, new_camera
