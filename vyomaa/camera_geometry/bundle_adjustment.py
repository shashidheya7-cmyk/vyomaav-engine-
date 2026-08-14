"""Non-linear Levenberg-Marquardt Bundle Adjustment solver using SciPy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import least_squares

from ..core.contracts import Camera
from ..core.exceptions import BundleAdjustmentError


@dataclass
class BundleAdjustmentResult:
    """Convergence and quality summary of bundle adjustment optimization."""

    is_converged: bool
    initial_reprojection_error_pixels: float
    final_reprojection_error_pixels: float
    optimized_cameras: List[Camera]
    optimized_points_3d: np.ndarray
    num_observations: int
    num_iterations: int


class BundleAdjuster:
    """Refines multi-view camera extrinsic poses and 3D landmark points by minimizing reprojection error."""

    @staticmethod
    def _rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
        theta = np.linalg.norm(rvec)
        if theta < 1e-8:
            return np.eye(3, dtype=np.float32)
        k = rvec / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float32)
        return (np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)).astype(np.float32)

    @staticmethod
    def _matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
        cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if theta < 1e-8:
            return np.zeros(3, dtype=np.float32)
        r = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float32)
        return (r / (2.0 * np.sin(theta)) * theta).astype(np.float32)

    @classmethod
    def optimize(
        cls,
        cameras: List[Camera],
        points_3d: np.ndarray,  # (N, 3)
        camera_indices: np.ndarray,  # (M,)
        point_indices: np.ndarray,   # (M,)
        points_2d: np.ndarray,       # (M, 2)
        max_nfev: int = 50,
    ) -> BundleAdjustmentResult:
        """Execute non-linear least squares bundle adjustment."""
        num_cameras = len(cameras)
        num_points = len(points_3d)
        num_obs = len(points_2d)

        if num_obs == 0 or num_cameras == 0 or num_points == 0:
            raise BundleAdjustmentError("Insufficient observations for bundle adjustment")

        # Parameter vector: 6 params per camera [rx, ry, rz, tx, ty, tz] + 3 params per point [x, y, z]
        cam_params = np.zeros((num_cameras, 6), dtype=np.float32)
        intrinsics = []
        for i, c in enumerate(cameras):
            R = c.RT[:3, :3]
            t = c.RT[:3, 3]
            rvec = cls._matrix_to_rodrigues(R)
            cam_params[i, :3] = rvec
            cam_params[i, 3:] = t
            intrinsics.append((c.focal_length_x, c.focal_length_y, c.principal_point_x, c.principal_point_y))

        initial_params = np.hstack([cam_params.ravel(), points_3d.ravel()])

        def _project(params: np.ndarray) -> np.ndarray:
            c_params = params[: num_cameras * 6].reshape((num_cameras, 6))
            p_3d = params[num_cameras * 6 :].reshape((num_points, 3))

            pts = p_3d[point_indices]
            c_idx = camera_indices

            rvecs = c_params[c_idx, :3]
            tvecs = c_params[c_idx, 3:]

            # Rotate & translate
            # Simplified vectorized point transformation
            projected = np.zeros((num_obs, 2), dtype=np.float32)
            for j in range(num_obs):
                cam_i = camera_indices[j]
                pt_i = point_indices[j]
                R_mat = cls._rodrigues_to_matrix(c_params[cam_i, :3])
                t_vec = c_params[cam_i, 3:]
                p_cam = R_mat @ p_3d[pt_i] + t_vec

                fx, fy, cx, cy = intrinsics[cam_i]
                if p_cam[2] > 1e-4:
                    projected[j, 0] = fx * (p_cam[0] / p_cam[2]) + cx
                    projected[j, 1] = fy * (p_cam[1] / p_cam[2]) + cy
                else:
                    projected[j] = [cx, cy]

            return (projected - points_2d).ravel()

        init_residuals = _project(initial_params)
        init_rmse = float(np.sqrt(np.mean(init_residuals ** 2)))

        res = least_squares(
            _project,
            initial_params,
            method="trf",
            loss="huber",
            ftol=1e-4,
            xtol=1e-4,
            max_nfev=max_nfev,
        )

        final_rmse = float(np.sqrt(np.mean(res.fun ** 2)))
        opt_cam_params = res.x[: num_cameras * 6].reshape((num_cameras, 6))
        opt_points_3d = res.x[num_cameras * 6 :].reshape((num_points, 3)).astype(np.float32)

        # Update cameras
        optimized_cams: List[Camera] = []
        for i, c in enumerate(cameras):
            R_opt = cls._rodrigues_to_matrix(opt_cam_params[i, :3])
            t_opt = opt_cam_params[i, 3:]
            RT_opt = np.eye(4, dtype=np.float32)
            RT_opt[:3, :3] = R_opt
            RT_opt[:3, 3] = t_opt

            opt_c = Camera.from_matrices(c.K, RT_opt, (c.image_width, c.image_height), name=f"{c.name}_opt")
            optimized_cams.append(opt_c)

        return BundleAdjustmentResult(
            is_converged=bool(res.success),
            initial_reprojection_error_pixels=round(init_rmse, 3),
            final_reprojection_error_pixels=round(final_rmse, 3),
            optimized_cameras=optimized_cams,
            optimized_points_3d=opt_points_3d,
            num_observations=num_obs,
            num_iterations=int(res.nfev),
        )
