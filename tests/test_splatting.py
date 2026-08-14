"""
VYOMAAV Base Model Engine
Test Suite: tests/test_splatting.py

Pytest suite validating Sprint 10: SOMG-to-3D-Gaussian conversion, anisotropic cloud attributes,
and differentiable depth-sorted alpha-blended rasterization.
"""

import pytest
import torch
from somg.scene import SceneState
from somg.entity import SOMGEntity, SpatialComponent, MaterialComponent
from somg.camera import CameraFrame
from engine.splatting import (
    GaussianSplatCloud, SOMGToGaussianSplatConverter, DifferentiableSplatRasterizer
)


def test_somg_entity_to_gaussian_splat_conversion():
    entity = SOMGEntity(
        entity_id="chair_01",
        spatial=SpatialComponent(bbox_min=[-1.0, 0.0, -1.0], bbox_max=[1.0, 2.0, 1.0]),
        material=MaterialComponent(material_type="wood")
    )

    splats = SOMGToGaussianSplatConverter.generate_entity_gaussians(
        entity, points_per_entity=50, device=torch.device("cpu")
    )

    assert isinstance(splats, GaussianSplatCloud)
    assert splats.num_splats() == 50
    assert splats.means.shape == (50, 3)
    assert splats.scales.shape == (50, 3)
    assert splats.rotations.shape == (50, 4)
    assert splats.opacities.shape == (50, 1)
    assert splats.colors.shape == (50, 3)

    # Assert points fall within bounding box bounds
    assert (splats.means[:, 0] >= -1.0).all() and (splats.means[:, 0] <= 1.0).all()
    assert (splats.means[:, 1] >= 0.0).all() and (splats.means[:, 1] <= 2.0).all()


def test_scene_to_gaussian_splat_cloud():
    scene = SceneState(scene_id="Office")
    e1 = SOMGEntity("desk", spatial=SpatialComponent(bbox_min=[-2.0, 0.0, -1.0], bbox_max=[2.0, 1.0, 1.0]))
    e2 = SOMGEntity("lamp", spatial=SpatialComponent(bbox_min=[0.0, 1.0, 0.0], bbox_max=[0.5, 1.8, 0.5]))

    scene.base_graph.add_node(e1)
    scene.base_graph.add_node(e2)

    cloud = SOMGToGaussianSplatConverter.convert_scene(scene, points_per_entity=30)

    assert cloud.num_splats() == 60  # 30 * 2 entities


def test_differentiable_splat_rasterizer_rendering():
    # Setup Splat Cloud with single bright red Gaussian at z = 3.0
    means = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32)
    scales = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    opacities = torch.tensor([[0.9]], dtype=torch.float32)
    colors = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)  # Red

    cloud = GaussianSplatCloud(means, scales, rotations, opacities, colors)

    # Camera looking forward at origin
    camera = CameraFrame(
        frame_id="cam_0",
        pose_se3=[1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0],  # Identity pose
        intrinsics_k=[150.0, 0.0, 32.0,
                      0.0, 150.0, 32.0,
                      0.0, 0.0, 1.0]     # Center at (32, 32)
    )

    rasterizer = DifferentiableSplatRasterizer(image_width=64, image_height=64)
    rendered_img = rasterizer(cloud, camera)

    assert rendered_img.shape == (3, 64, 64)
    # Check that center pixel has significant red channel value
    center_red = rendered_img[0, 32, 32].item()
    assert center_red > 0.5