"""
VYOMAAV Base Model Engine
Module: engine.splatting

3D Gaussian Splatting Visual Layer Engine.
Converts SOMG entities and scene graphs into parameterized 3D Gaussian primitives
(Positions mu, Scales s, Rotations q, Opacities alpha, Base Colors c) and executes
depth-sorted tile-based alpha-blended rasterization.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import math
import torch
import torch.nn as nn

from somg.scene import SceneState
from somg.entity import SOMGEntity
from somg.camera import CameraFrame


@dataclass
class GaussianSplatCloud:
    """Tensor container for N 3D Gaussian primitives.

    Tensors:
        means: (N, 3) -> Centers mu in world space
        scales: (N, 3) -> Log/linear anisotropic scale factors s
        rotations: (N, 4) -> Unit quaternions q = (w, x, y, z)
        opacities: (N, 1) -> Sigmoidal opacity values alpha in [0, 1]
        colors: (N, 3) -> Base RGB colors in [0, 1]
    """
    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor

    def num_splats(self) -> int:
        return self.means.shape[0]


class SOMGToGaussianSplatConverter:
    """Converts Spatial Object Memory Graph (SOMG) scene states into GaussianSplatClouds."""

    @staticmethod
    def generate_entity_gaussians(
        entity: SOMGEntity,
        points_per_entity: int = 100,
        device: torch.device = torch.device("cpu")
    ) -> GaussianSplatCloud:
        """Populates a 3D Gaussian ellipsoid within an entity's bounding box."""
        b_min = torch.tensor(entity.spatial.bbox_min, dtype=torch.float32, device=device)
        b_max = torch.tensor(entity.spatial.bbox_max, dtype=torch.float32, device=device)

        center = (b_min + b_max) / 2.0
        extents = torch.clamp((b_max - b_min) / 2.0, min=0.05)

        # Uniform random point distribution within bounding box volume
        rand_offsets = (torch.rand((points_per_entity, 3), device=device) * 2.0 - 1.0)
        means = center.unsqueeze(0) + rand_offsets * extents.unsqueeze(0)

        # Gaussian scale relative to entity size
        splat_scale = (extents / math.sqrt(points_per_entity)).unsqueeze(0).repeat(points_per_entity, 1)
        scales = torch.clamp(splat_scale, min=0.01)

        # Identity quaternions [w=1, x=0, y=0, z=0]
        rotations = torch.zeros((points_per_entity, 4), device=device)
        rotations[:, 0] = 1.0

        # Default opacity
        opacities = torch.full((points_per_entity, 1), 0.85, device=device)

        # Material-derived RGB base color assignment
        color_map = {
            "wood": torch.tensor([0.55, 0.35, 0.15], device=device),
            "fabric": torch.tensor([0.20, 0.40, 0.80], device=device),
            "glass": torch.tensor([0.80, 0.90, 0.95], device=device),
            "metal": torch.tensor([0.70, 0.70, 0.75], device=device),
            "generic": torch.tensor([0.60, 0.60, 0.60], device=device)
        }
        base_color = color_map.get(entity.material.material_type, color_map["generic"])
        # Add slight color jitter
        jitter = (torch.rand((points_per_entity, 3), device=device) - 0.5) * 0.1
        colors = torch.clamp(base_color.unsqueeze(0) + jitter, 0.0, 1.0)

        return GaussianSplatCloud(means, scales, rotations, opacities, colors)

    @classmethod
    def convert_scene(
        cls,
        scene: SceneState,
        points_per_entity: int = 100,
        device: torch.device = torch.device("cpu")
    ) -> GaussianSplatCloud:
        """Converts an entire resolved SceneState graph into a unified GaussianSplatCloud."""
        graph = scene.resolve_active_graph()
        all_means, all_scales, all_rots, all_opacs, all_cols = [], [], [], [], []

        for entity in graph.nodes.values():
            splats = cls.generate_entity_gaussians(entity, points_per_entity, device)
            all_means.append(splats.means)
            all_scales.append(splats.scales)
            all_rots.append(splats.rotations)
            all_opacs.append(splats.opacities)
            all_cols.append(splats.colors)

        if not all_means:
            # Empty cloud fallback
            return GaussianSplatCloud(
                torch.zeros((0, 3), device=device),
                torch.zeros((0, 3), device=device),
                torch.zeros((0, 4), device=device),
                torch.zeros((0, 1), device=device),
                torch.zeros((0, 3), device=device)
            )

        return GaussianSplatCloud(
            torch.cat(all_means, dim=0),
            torch.cat(all_scales, dim=0),
            torch.cat(all_rots, dim=0),
            torch.cat(all_opacs, dim=0),
            torch.cat(all_cols, dim=0)
        )


class DifferentiableSplatRasterizer(nn.Module):
    """Differentiable depth-sorted tile alpha-blending rasterizer for 3D Gaussian Splats."""

    def __init__(self, image_width: int = 256, image_height: int = 256):
        super().__init__()
        self.width = image_width
        self.height = image_height

    def forward(
        self,
        splats: GaussianSplatCloud,
        camera_frame: CameraFrame,
        bg_color: torch.Tensor = None
    ) -> torch.Tensor:
        """Renders 3D Gaussian Splat Cloud to a 2D RGB image tensor (C, H, W).

        Alpha Blending Equation:
            C = sum_{i in N} c_i * alpha_i * prod_{j=1}^{i-1} (1 - alpha_j)
        """
        device = splats.means.device
        if bg_color is None:
            bg_color = torch.tensor([0.05, 0.05, 0.05], device=device)

        if splats.num_splats() == 0:
            return bg_color.view(3, 1, 1).expand(3, self.height, self.width)

        # 1. Unpack Camera Pose (12-element SE3 matrix: 3x4 [R | t])
        pose_mat = torch.tensor(camera_frame.pose_se3, dtype=torch.float32, device=device).view(3, 4)
        R_cam = pose_mat[:, :3]
        t_cam = pose_mat[:, 3]

        # Camera Intrinsics K (3x3)
        K_mat = torch.tensor(camera_frame.intrinsics_k, dtype=torch.float32, device=device).view(3, 3)
        f_x, f_y = K_mat[0, 0], K_mat[1, 1]
        c_x, c_y = K_mat[0, 2], K_mat[1, 2]

        # 2. Transform Gaussian centers mu to Camera Space: mu_cam = R * mu + t
        means_cam = torch.matmul(splats.means, R_cam.T) + t_cam.unsqueeze(0)  # (N, 3)

        # 3. Filter points behind near plane (z <= 0.1)
        z_mask = means_cam[:, 2] > 0.1
        if not z_mask.any():
            return bg_color.view(3, 1, 1).expand(3, self.height, self.width)

        means_valid = means_cam[z_mask]
        colors_valid = splats.colors[z_mask]
        opacs_valid = splats.opacities[z_mask]
        scales_valid = splats.scales[z_mask]

        # 4. Project centers to 2D Pixel Coordinates
        z_depth = means_valid[:, 2]
        u_px = (means_valid[:, 0] * f_x / z_depth) + c_x
        v_px = (means_valid[:, 1] * f_y / z_depth) + c_y

        # Scale 2D screen footprint proportional to focal length and scale
        radius_px = torch.clamp((scales_valid.mean(dim=1) * f_x / z_depth), min=1.5, max=50.0)

        # 5. Front-to-Back Depth Sorting
        sorted_indices = torch.argsort(z_depth, descending=False)

        u_sorted = u_px[sorted_indices]
        v_sorted = v_px[sorted_indices]
        r_sorted = radius_px[sorted_indices]
        c_sorted = colors_valid[sorted_indices]
        a_sorted = opacs_valid[sorted_indices]

        # 6. Render Pixel Grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32, device=device),
            torch.arange(self.width, dtype=torch.float32, device=device),
            indexing="ij"
        )  # (H, W)

        rendered_image = bg_color.view(3, 1, 1).expand(3, self.height, self.width).clone()
        transmittance = torch.ones((self.height, self.width), device=device)

        # Accumulate splats in front-to-back order
        num_valid = u_sorted.shape[0]
        max_render_splats = min(num_valid, 200)  # Cap rendering loop for high FPS

        for i in range(max_render_splats):
            cx, cy, radius = u_sorted[i], v_sorted[i], r_sorted[i]
            col = c_sorted[i]
            base_alpha = a_sorted[i]

            # Gaussian Radial Falloff: G(r) = exp(-0.5 * (dist^2 / radius^2))
            dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
            splat_alpha = base_alpha * torch.exp(-0.5 * dist_sq / (radius ** 2 + 1e-5))
            splat_alpha = torch.clamp(splat_alpha, 0.0, 0.95)

            # Mask out splats further than 3 * radius
            mask = dist_sq <= (3.0 * radius) ** 2
            effective_alpha = splat_alpha * mask.float()

            # Front-to-back alpha blending weight
            weight = effective_alpha * transmittance
            rendered_image = rendered_image + weight.unsqueeze(0) * col.view(3, 1, 1)
            transmittance = transmittance * (1.0 - effective_alpha)

            # Early ray termination if fully opaque
            if (transmittance < 0.01).all():
                break

        return torch.clamp(rendered_image, 0.0, 1.0)