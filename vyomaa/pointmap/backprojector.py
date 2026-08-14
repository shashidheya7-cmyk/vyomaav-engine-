"""Dense pixel-to-world back-projection converting depth and camera poses into canonical 3D PointClouds."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from ..core.contracts import Camera, DepthMap, Observation
from ..core.exceptions import GeometryError
from ..core.provenance import ProvenanceRecord
from ..representations.point_cloud import PointCloud


class DepthBackprojector:
    """Computes exact metric camera-space and world-space 3D point coordinates."""

    @staticmethod
    def backproject_observation(
        observation: Observation,
        depth_array: np.ndarray,
        image_rgb: Optional[np.ndarray] = None,
        normals_array: Optional[np.ndarray] = None,
        stride: int = 1,
        min_depth: float = 0.05,
        max_depth: float = 100.0,
    ) -> PointCloud:
        """Unproject dense 2D pixel coordinates (u, v) and depth Z to 3D world points."""
        if observation.camera is None:
            raise GeometryError("Observation must contain calibrated Camera contract for back-projection")

        cam = observation.camera
        fx, fy = cam.focal_length_x, cam.focal_length_y
        cx, cy = cam.principal_point_x, cam.principal_point_y

        h, w = depth_array.shape[:2]

        # Generate meshgrid of pixel coordinates
        u_coords, v_coords = np.meshgrid(
            np.arange(0, w, stride, dtype=np.float32),
            np.arange(0, h, stride, dtype=np.float32),
        )

        z = depth_array[::stride, ::stride].astype(np.float32)

        # Depth validity mask
        valid_mask = (z >= min_depth) & (z <= max_depth) & np.isfinite(z)
        if not np.any(valid_mask):
            return PointCloud(
                name=f"Empty_PC_{observation.frame_id}",
                points=np.empty((0, 3), dtype=np.float32),
                provenance=ProvenanceRecord(producer_subsystem="pointmap"),
            )

        u_valid = u_coords[valid_mask]
        v_valid = v_coords[valid_mask]
        z_valid = z[valid_mask]

        # Camera-space coordinates
        x_cam = (u_valid - cx) * z_valid / fx
        y_cam = (v_valid - cy) * z_valid / fy
        z_cam = z_valid

        pts_cam = np.column_stack([x_cam, y_cam, z_cam])  # (N, 3)

        # Transform to world space: P_world = R_w2c^T * (P_cam - t_w2c)
        R_w2c = cam.RT[:3, :3]
        t_w2c = cam.RT[:3, 3]

        R_c2w = R_w2c.T
        pts_world = (pts_cam - t_w2c) @ R_w2c  # Equal to (R_w2c.T @ (pts_cam - t_w2c).T).T

        # Extract RGB colors
        colors = None
        if image_rgb is not None:
            rgb_sub = image_rgb[::stride, ::stride]
            colors = rgb_sub[valid_mask].astype(np.float32)
            if colors.max() > 1.0:
                colors /= 255.0

        # Transform normals if present
        normals = None
        if normals_array is not None:
            norm_sub = normals_array[::stride, ::stride]
            norm_cam = norm_sub[valid_mask]
            normals = (norm_cam @ R_w2c).astype(np.float32)

        confidence = np.ones(len(pts_world), dtype=np.float32) * observation.confidence_score

        pc = PointCloud(
            name=f"PointCloud_{observation.frame_id}",
            points=pts_world.astype(np.float32),
            normals=normals,
            colors=colors,
            confidence_values=confidence,
            confidence_score=observation.confidence_score,
            provenance=ProvenanceRecord(
                producer_subsystem="pointmap",
                parent_artifact_ids=[observation.artifact_id],
                generation_parameters={"stride": stride, "point_count": len(pts_world)},
            ),
        )
        return pc

    @staticmethod
    def merge_point_clouds(clouds: List[PointCloud], voxel_size: float = 0.01) -> PointCloud:
        """Merge multiple PointClouds with spatial voxel grid filtering."""
        if not clouds:
            return PointCloud(points=np.empty((0, 3), dtype=np.float32))

        all_pts, all_normals, all_colors, all_confs, parent_ids = [], [], [], [], []
        for c in clouds:
            if len(c.points) > 0:
                all_pts.append(c.points)
                parent_ids.append(c.artifact_id)
                if c.normals is not None:
                    all_normals.append(c.normals)
                if c.colors is not None:
                    all_colors.append(c.colors)
                if c.confidence_values is not None:
                    all_confs.append(c.confidence_values)

        if not all_pts:
            return PointCloud(points=np.empty((0, 3), dtype=np.float32))

        merged_pts = np.vstack(all_pts)
        merged_normals = np.vstack(all_normals) if len(all_normals) == len(clouds) else None
        merged_colors = np.vstack(all_colors) if len(all_colors) == len(clouds) else None
        merged_confs = np.concatenate(all_confs) if len(all_confs) == len(clouds) else None

        # Simple spatial voxel downsampling
        if voxel_size > 0:
            coords = np.floor(merged_pts / voxel_size).astype(np.int64)
            _, unique_indices = np.unique(coords, axis=0, return_index=True)
            merged_pts = merged_pts[unique_indices]
            if merged_normals is not None:
                merged_normals = merged_normals[unique_indices]
            if merged_colors is not None:
                merged_colors = merged_colors[unique_indices]
            if merged_confs is not None:
                merged_confs = merged_confs[unique_indices]

        return PointCloud(
            name="Merged World PointCloud",
            points=merged_pts,
            normals=merged_normals,
            colors=merged_colors,
            confidence_values=merged_confs,
            provenance=ProvenanceRecord(
                producer_subsystem="pointmap",
                parent_artifact_ids=parent_ids,
            ),
        )
