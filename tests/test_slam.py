"""
VYOMAAV Base Model Engine
Test Suite: tests/test_slam.py

Pytest suite validating Sprint 19: IMU pre-integration, feature keypoint extraction,
keyframe selection, and pose graph optimization over SOMG scenes.
"""

import pytest
import torch
from somg.scene import SceneState
from engine.slam import VisualInertialSLAMEngine, IMUSample, SLAMKeyframe


def test_imu_pre_integration():
    scene = SceneState(scene_id="SLAMWorld")
    engine = VisualInertialSLAMEngine(scene)

    # Constant forward acceleration 1.0 m/s^2 along Z axis
    imu = IMUSample(timestamp_s=0.01, accel=[0.0, 0.0, 1.0], gyro=[0.0, 0.0, 0.0])
    engine.integrate_imu(imu, dt=0.1)

    assert engine.velocity[2] == 0.1  # v = a * dt = 1.0 * 0.1
    assert engine.current_pose_se3[11] > 0.0  # Pos z > 0


def test_slam_keyframe_spawning_and_feature_detection():
    scene = SceneState(scene_id="SLAMWorld")
    engine = VisualInertialSLAMEngine(scene, keyframe_trans_threshold=0.1)

    dummy_image = torch.zeros((3, 256, 256))
    intrinsics = [100.0, 0.0, 128.0, 0.0, 100.0, 128.0, 0.0, 0.0, 1.0]

    # Frame 1: Should spawn initial keyframe 0
    f1 = engine.process_frame(timestamp_s=0.0, rgb_image_tensor=dummy_image, intrinsics_k=intrinsics)
    assert len(engine.keyframes) == 1
    assert engine.keyframes[0].keyframe_id == 0

    # Simulate translation past threshold
    engine.current_pose_se3[11] += 0.25

    # Frame 2: Should spawn keyframe 1
    f2 = engine.process_frame(timestamp_s=0.033, rgb_image_tensor=dummy_image, intrinsics_k=intrinsics)
    assert len(engine.keyframes) == 2
    assert engine.keyframes[1].keyframe_id == 1

    # Verify SOMG Camera Graph updated
    cam_frame = scene.camera_graph.get_frame(f2.frame_id)
    assert cam_frame is not None
    assert cam_frame.pose_se3[11] == engine.current_pose_se3[11]