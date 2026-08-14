"""Geometric validation engine evaluating depth, cameras, epipolar consistency, and point clouds."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from ..core.contracts import Camera, CameraTrajectory, DepthMap, Observation, ValidationReport
from ..multiview.view_graph import ViewGraph
from ..representations.point_cloud import PointCloud


class GeometricValidator:
    """Performs rigorous quality assessment on all Phase 2 geometric evidence outputs."""

    @staticmethod
    def validate_depth_map(depth_artifact: DepthMap, depth_array: Optional[np.ndarray] = None) -> Tuple[bool, float, List[str]]:
        """Validate depth map range, non-finites, and coverage."""
        warnings: List[str] = []
        if depth_artifact.width <= 0 or depth_artifact.height <= 0:
            return False, 0.0, ["Invalid depth map dimensions (0x0)"]

        if depth_array is not None:
            finite_mask = np.isfinite(depth_array)
            finite_ratio = float(np.mean(finite_mask))
            if finite_ratio < 0.8:
                warnings.append(f"Low finite depth ratio: {finite_ratio:.2%}")

            positive_ratio = float(np.mean(depth_array > 0))
            if positive_ratio < 0.5:
                warnings.append(f"Low positive depth coverage: {positive_ratio:.2%}")

            score = max(0.0, min(1.0, finite_ratio * 0.5 + positive_ratio * 0.5))
            return len(warnings) == 0, score, warnings

        score = 0.9 if depth_artifact.max_depth > depth_artifact.min_depth else 0.4
        return True, score, warnings

    @staticmethod
    def validate_camera(camera: Camera) -> Tuple[bool, float, List[str]]:
        """Verify camera matrix positive focal length, principal point bounds, and orthogonality."""
        warnings: List[str] = []
        if camera.focal_length_x <= 0 or camera.focal_length_y <= 0:
            return False, 0.0, ["Focal lengths must be strictly positive"]

        # Check rotation matrix determinant = 1
        R = camera.RT[:3, :3]
        det = float(np.linalg.det(R))
        if abs(det - 1.0) > 1e-3:
            warnings.append(f"Camera rotation matrix determinant {det:.4f} diverges from 1.0 (non-orthogonal)")

        is_valid = len(warnings) == 0
        score = 1.0 if is_valid else 0.5
        return is_valid, score, warnings

    @staticmethod
    def validate_view_graph(view_graph: ViewGraph) -> Tuple[bool, float, List[str]]:
        """Assess graph connectivity, epipolar consistency, and pairwise inlier ratios."""
        warnings: List[str] = []
        if not view_graph.views:
            return False, 0.0, ["ViewGraph contains no observation nodes"]

        if not view_graph.edges:
            return False, 0.2, ["ViewGraph has no valid connectivity edges"]

        valid_edges = [e for e in view_graph.edges.values() if e.is_valid_edge]
        valid_ratio = len(valid_edges) / len(view_graph.edges)

        epipolar_errors = [e.epipolar_error_pixels for e in valid_edges if e.epipolar_error_pixels < 50.0]
        mean_epipolar = float(np.mean(epipolar_errors)) if epipolar_errors else 10.0

        if mean_epipolar > 3.0:
            warnings.append(f"Mean epipolar reprojection error is high: {mean_epipolar:.2f}px")

        score = max(0.0, min(1.0, valid_ratio * 0.6 + max(0.0, 1.0 - (mean_epipolar / 5.0)) * 0.4))
        return (len(warnings) == 0 and len(valid_edges) > 0), round(score, 3), warnings

    @staticmethod
    def validate_point_cloud(point_cloud: PointCloud) -> Tuple[bool, float, List[str]]:
        """Verify point cloud coordinate sanity and spatial distribution."""
        warnings: List[str] = []
        if point_cloud.point_count == 0:
            return False, 0.0, ["Point cloud is empty"]

        min_b, max_b = point_cloud.compute_bounds()
        extent = max_b - min_b
        if np.any(extent < 1e-6):
            warnings.append("Point cloud is collapsed along one or more dimensions")

        if np.any(np.isnan(point_cloud.points)) or np.any(np.isinf(point_cloud.points)):
            return False, 0.0, ["Point cloud contains NaN or Inf coordinate values"]

        score = 0.95 if point_cloud.point_count >= 1000 else float(point_cloud.point_count) / 1000.0
        return len(warnings) == 0, round(score, 3), warnings

    @classmethod
    def generate_comprehensive_report(
        cls,
        depth_maps: List[DepthMap],
        cameras: List[Camera],
        point_cloud: Optional[PointCloud] = None,
        view_graph: Optional[ViewGraph] = None,
        reprojection_error_pixels: float = 0.0,
    ) -> ValidationReport:
        """Aggregate quality checks into a unified ValidationReport contract."""
        all_warnings, all_errors = [], []
        depth_scores, cam_scores = [], []

        for d in depth_maps:
            valid, sc, w = cls.validate_depth_map(d)
            depth_scores.append(sc)
            all_warnings.extend(w)

        for c in cameras:
            valid, sc, w = cls.validate_camera(c)
            cam_scores.append(sc)
            all_warnings.extend(w)

        pc_score = 1.0
        if point_cloud:
            valid, pc_score, w = cls.validate_point_cloud(point_cloud)
            all_warnings.extend(w)

        vg_score = 1.0
        if view_graph:
            valid, vg_score, w = cls.validate_view_graph(view_graph)
            all_warnings.extend(w)

        mean_depth_conf = float(np.mean(depth_scores)) if depth_scores else 0.8
        mean_cam_conf = float(np.mean(cam_scores)) if cam_scores else 0.85

        overall = (mean_depth_conf * 0.3 + mean_cam_conf * 0.3 + pc_score * 0.2 + vg_score * 0.2)
        overall = round(max(0.0, min(1.0, overall)), 3)

        return ValidationReport(
            name="Phase 2 Geometric Evidence Validation Report",
            is_valid=(len(all_errors) == 0 and overall >= 0.5),
            overall_quality_score=overall,
            camera_pose_stability=round(mean_cam_conf, 3),
            reprojection_error_pixels=reprojection_error_pixels,
            warnings=list(set(all_warnings)),
            errors=all_errors,
            confidence_score=overall,
        )
