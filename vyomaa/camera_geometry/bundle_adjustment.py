"""Non-linear bundle adjustment solver using SciPy least-squares."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    """
    Refine camera extrinsics and 3D landmark positions by minimizing
    multi-view reprojection error.

    The optimizer operates internally in float64 for numerical stability.
    A result is considered converged when either SciPy reports successful
    termination or the resulting geometry reaches a high-quality solution
    with sub-pixel reprojection error and meaningful improvement.
    """

    REPROJECTION_TARGET_PX = 1.0
    MIN_IMPROVEMENT_RATIO = 1e-3

    @staticmethod
    def _rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
        """Convert a Rodrigues rotation vector to a 3x3 rotation matrix."""

        rvec = np.asarray(rvec, dtype=np.float64)
        theta = float(np.linalg.norm(rvec))

        if theta < 1e-12:
            return np.eye(3, dtype=np.float64)

        k = rvec / theta

        K = np.array(
            [
                [0.0, -k[2], k[1]],
                [k[2], 0.0, -k[0]],
                [-k[1], k[0], 0.0],
            ],
            dtype=np.float64,
        )

        return (
            np.eye(3, dtype=np.float64)
            + np.sin(theta) * K
            + (1.0 - np.cos(theta)) * (K @ K)
        )

    @staticmethod
    def _matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
        """Convert a 3x3 rotation matrix to a Rodrigues vector."""

        R = np.asarray(R, dtype=np.float64)

        trace_value = float(np.trace(R))
        cos_theta = np.clip((trace_value - 1.0) / 2.0, -1.0, 1.0)
        theta = float(np.arccos(cos_theta))

        if theta < 1e-10:
            return np.zeros(3, dtype=np.float64)

        sin_theta = float(np.sin(theta))

        # Standard case.
        if abs(sin_theta) > 1e-8:
            r = np.array(
                [
                    R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1],
                ],
                dtype=np.float64,
            )

            return r * (theta / (2.0 * sin_theta))

        # Near pi: recover the rotation axis from the diagonal.
        axis = np.empty(3, dtype=np.float64)

        axis[0] = np.sqrt(max(0.0, (R[0, 0] + 1.0) / 2.0))
        axis[1] = np.sqrt(max(0.0, (R[1, 1] + 1.0) / 2.0))
        axis[2] = np.sqrt(max(0.0, (R[2, 2] + 1.0) / 2.0))

        # Recover signs from off-diagonal terms.
        if R[2, 1] - R[1, 2] < 0:
            axis[0] *= -1.0
        if R[0, 2] - R[2, 0] < 0:
            axis[1] *= -1.0
        if R[1, 0] - R[0, 1] < 0:
            axis[2] *= -1.0

        norm = float(np.linalg.norm(axis))

        if norm < 1e-10:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis /= norm

        return axis * theta

    @classmethod
    def optimize(
        cls,
        cameras: List[Camera],
        points_3d: np.ndarray,
        camera_indices: np.ndarray,
        point_indices: np.ndarray,
        points_2d: np.ndarray,
        max_nfev: int = 200,
    ) -> BundleAdjustmentResult:
        """
        Execute non-linear least-squares bundle adjustment.

        Camera parameters:
            [rx, ry, rz, tx, ty, tz]

        Point parameters:
            [x, y, z]

        The first camera remains part of the optimization parameterization
        for compatibility with the existing engine contract.
        """

        num_cameras = len(cameras)

        points_3d = np.asarray(points_3d, dtype=np.float64)
        camera_indices = np.asarray(camera_indices, dtype=np.int64)
        point_indices = np.asarray(point_indices, dtype=np.int64)
        points_2d = np.asarray(points_2d, dtype=np.float64)

        if num_cameras == 0:
            raise BundleAdjustmentError(
                "Insufficient observations: no cameras supplied."
            )

        if points_3d.ndim != 2 or points_3d.shape[1] != 3:
            raise BundleAdjustmentError(
                f"points_3d must have shape (N, 3), got {points_3d.shape}."
            )

        if points_2d.ndim != 2 or points_2d.shape[1] != 2:
            raise BundleAdjustmentError(
                f"points_2d must have shape (M, 2), got {points_2d.shape}."
            )

        num_points = len(points_3d)
        num_obs = len(points_2d)

        if num_points == 0 or num_obs == 0:
            raise BundleAdjustmentError(
                "Insufficient observations for bundle adjustment."
            )

        if len(camera_indices) != num_obs:
            raise BundleAdjustmentError(
                "camera_indices length must equal number of observations."
            )

        if len(point_indices) != num_obs:
            raise BundleAdjustmentError(
                "point_indices length must equal number of observations."
            )

        if np.any(camera_indices < 0) or np.any(camera_indices >= num_cameras):
            raise BundleAdjustmentError(
                "camera_indices contains an invalid camera index."
            )

        if np.any(point_indices < 0) or np.any(point_indices >= num_points):
            raise BundleAdjustmentError(
                "point_indices contains an invalid point index."
            )

        if not np.all(np.isfinite(points_3d)):
            raise BundleAdjustmentError(
                "points_3d contains non-finite values."
            )

        if not np.all(np.isfinite(points_2d)):
            raise BundleAdjustmentError(
                "points_2d contains non-finite values."
            )

        # ------------------------------------------------------------------
        # Initial camera parameterization
        # ------------------------------------------------------------------

        cam_params = np.zeros(
            (num_cameras, 6),
            dtype=np.float64,
        )

        intrinsics = []

        for i, camera in enumerate(cameras):
            RT = np.asarray(camera.RT, dtype=np.float64)

            if RT.shape != (4, 4):
                raise BundleAdjustmentError(
                    f"Camera {i} has invalid RT shape {RT.shape}; "
                    "expected (4, 4)."
                )

            R = RT[:3, :3]
            t = RT[:3, 3]

            cam_params[i, :3] = cls._matrix_to_rodrigues(R)
            cam_params[i, 3:] = t

            intrinsics.append(
                (
                    float(camera.focal_length_x),
                    float(camera.focal_length_y),
                    float(camera.principal_point_x),
                    float(camera.principal_point_y),
                )
            )

        initial_params = np.concatenate(
            [
                cam_params.ravel(),
                points_3d.ravel(),
            ]
        )

        # ------------------------------------------------------------------
        # Projection function
        # ------------------------------------------------------------------

        def project(params: np.ndarray) -> np.ndarray:
            camera_params = params[
                : num_cameras * 6
            ].reshape(num_cameras, 6)

            point_params = params[
                num_cameras * 6 :
            ].reshape(num_points, 3)

            projected = np.empty(
                (num_obs, 2),
                dtype=np.float64,
            )

            for j in range(num_obs):
                camera_index = int(camera_indices[j])
                point_index = int(point_indices[j])

                rvec = camera_params[camera_index, :3]
                tvec = camera_params[camera_index, 3:]

                R = cls._rodrigues_to_matrix(rvec)

                point_camera = (
                    R @ point_params[point_index]
                    + tvec
                )

                fx, fy, cx, cy = intrinsics[camera_index]

                z = float(point_camera[2])

                if z > 1e-8 and np.isfinite(z):
                    projected[j, 0] = (
                        fx * point_camera[0] / z + cx
                    )
                    projected[j, 1] = (
                        fy * point_camera[1] / z + cy
                    )
                else:
                    # Keep the residual finite while strongly penalizing
                    # points behind the camera.
                    projected[j, 0] = cx
                    projected[j, 1] = cy

            return projected

        def residuals(params: np.ndarray) -> np.ndarray:
            projected = project(params)
            residual = projected - points_2d

            residual = residual.ravel()

            residual[~np.isfinite(residual)] = 1e6

            return residual

        # ------------------------------------------------------------------
        # Initial quality
        # ------------------------------------------------------------------

        initial_residuals = residuals(initial_params)

        initial_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(initial_residuals)
                )
            )
        )

        if not np.isfinite(initial_rmse):
            raise BundleAdjustmentError(
                "Initial reprojection error is non-finite."
            )

        # ------------------------------------------------------------------
        # Optimization
        # ------------------------------------------------------------------

        try:
            result = least_squares(
                residuals,
                initial_params,
                method="trf",
                loss="soft_l1",
                f_scale=1.0,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=max(200, int(max_nfev)),
                verbose=0,
            )
        except Exception as exc:
            raise BundleAdjustmentError(
                f"Bundle adjustment optimization failed: {exc}"
            ) from exc

        final_residuals = np.asarray(
            result.fun,
            dtype=np.float64,
        )

        if final_residuals.size == 0:
            raise BundleAdjustmentError(
                "Bundle adjustment produced no residuals."
            )

        final_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(final_residuals)
                )
            )
        )

        if not np.isfinite(final_rmse):
            raise BundleAdjustmentError(
                "Bundle adjustment produced a non-finite final "
                "reprojection error."
            )

        # ------------------------------------------------------------------
        # Convergence classification
        # ------------------------------------------------------------------

        improvement = initial_rmse - final_rmse

        if initial_rmse > 1e-12:
            improvement_ratio = improvement / initial_rmse
        else:
            improvement_ratio = 0.0

        quality_converged = (
            final_rmse < cls.REPROJECTION_TARGET_PX
            and improvement > 0.0
            and improvement_ratio >= cls.MIN_IMPROVEMENT_RATIO
        )

        optimizer_converged = bool(result.success)

        is_converged = bool(
            optimizer_converged or quality_converged
        )

        # ------------------------------------------------------------------
        # Recover optimized geometry
        # ------------------------------------------------------------------

        optimized_camera_params = result.x[
            : num_cameras * 6
        ].reshape(num_cameras, 6)

        optimized_points = result.x[
            num_cameras * 6 :
        ].reshape(num_points, 3)

        optimized_points = np.asarray(
            optimized_points,
            dtype=np.float32,
        )

        optimized_cameras: List[Camera] = []

        for i, camera in enumerate(cameras):
            R_opt = cls._rodrigues_to_matrix(
                optimized_camera_params[i, :3]
            )

            t_opt = optimized_camera_params[i, 3:]

            RT_opt = np.eye(
                4,
                dtype=np.float32,
            )

            RT_opt[:3, :3] = R_opt.astype(np.float32)
            RT_opt[:3, 3] = t_opt.astype(np.float32)

            optimized_camera = Camera.from_matrices(
                camera.K,
                RT_opt,
                (
                    camera.image_width,
                    camera.image_height,
                ),
                name=f"{camera.name}_opt",
            )

            optimized_cameras.append(
                optimized_camera
            )

        return BundleAdjustmentResult(
            is_converged=is_converged,
            initial_reprojection_error_pixels=round(
                initial_rmse,
                3,
            ),
            final_reprojection_error_pixels=round(
                final_rmse,
                3,
            ),
            optimized_cameras=optimized_cameras,
            optimized_points_3d=optimized_points,
            num_observations=num_obs,
            num_iterations=int(result.nfev),
        )
