"""
VYOMAAV Base Model Engine
Test Suite: tests/test_spark_client.py

Pytest suite validating Sprint 14: WASD locomotion, gravity/jump integration,
flight mode toggles, bounding box collision pushback, and camera SE(3) frame generation.
"""

import pytest
from client.spark import (
    WebGPUSparkPlayerController, PlayerInputState, PlayerMovementMode
)


def test_player_controller_wasd_walk_locomotion():
    controller = WebGPUSparkPlayerController(initial_position=[0.0, 1.7, 0.0])

    # Move forward (W key) for 1 second
    input_w = PlayerInputState(move_forward=True)
    state = controller.update(input_w, dt=1.0)

    assert state.mode == PlayerMovementMode.WALK
    # Facing 0 degrees yaw (forward is -Z axis)
    assert state.position[2] < 0.0
    assert abs(state.position[0]) < 1e-4


def test_player_controller_jump_physics():
    controller = WebGPUSparkPlayerController(initial_position=[0.0, 1.7, 0.0])

    # Press Space to jump
    input_jump = PlayerInputState(jump_pressed=True)
    state_air = controller.update(input_jump, dt=0.1)

    assert state_air.mode == PlayerMovementMode.JUMP
    assert not state_air.is_grounded
    assert state_air.position[1] > 1.7  # Moved upward

    # Integrate gravity until landing
    input_idle = PlayerInputState()
    for _ in range(20):
        state_land = controller.update(input_idle, dt=0.1)

    assert state_land.is_grounded
    assert state_land.mode == PlayerMovementMode.WALK
    assert state_land.position[1] == 1.7  # Landed at ground level


def test_player_controller_flight_mode_toggle():
    controller = WebGPUSparkPlayerController(initial_position=[0.0, 1.7, 0.0])

    # Toggle Fly Mode (F key)
    input_fly = PlayerInputState(fly_toggle=True, move_up=True)
    state_fly = controller.update(input_fly, dt=1.0)

    assert state_fly.mode == PlayerMovementMode.FLY
    assert not state_fly.is_grounded
    assert state_fly.position[1] > 1.7  # Flew upward


def test_player_controller_bounding_box_collision_pushback():
    controller = WebGPUSparkPlayerController(initial_position=[0.0, 1.7, 0.0])

    # Obstacle Box at x in [0.8, 2.0], z in [-1.0, 1.0]
    boxes = [([0.8, 0.0, -1.0], [2.0, 3.0, 1.0])]

    # Attempt to move right into the box
    input_right = PlayerInputState(move_right=True)
    state = controller.update(input_right, dt=1.0, collision_boxes=boxes)

    # Verify player was pushed back outside player_radius (0.4m) from box_min (0.8m) -> x <= 0.4m
    assert state.position[0] <= 0.401


def test_get_camera_frame_and_spark_html_generation():
    controller = WebGPUSparkPlayerController(initial_position=[0.0, 2.0, -5.0])
    frame = controller.get_camera_frame("player_cam")

    assert frame.frame_id == "player_cam"
    assert len(frame.pose_se3) == 12
    assert frame.pose_se3[3] == 0.0  # px
    assert frame.pose_se3[7] == 2.0  # py
    assert frame.pose_se3[11] == -5.0  # pz

    html_code = WebGPUSparkPlayerController.generate_webgpu_spark_client_html("TestScene")
    assert "<!DOCTYPE html>" in html_code
    assert "VYOMAAV Spark Engine: TestScene" in html_code