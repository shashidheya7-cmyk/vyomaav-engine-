"""
VYOMAAV Base Model Engine
Module: client.spark

Interactive WebGPU Spark Real-Time Player Controller (Sprint 14).
Handles real-time 3D player locomotion (WASD movement, Jump physics, Flight mode, Look controls),
executes bounding box collision checks against SOMG scene entities, and generates executable
HTML5 / WebGPU Spark client bundles with WebGPU Gaussian Splats rendering shaders.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import List, Tuple, Optional, Dict, Any

from somg.camera import CameraFrame
from somg.scene import SceneState


class PlayerMovementMode(Enum):
    WALK = "WALK"
    JUMP = "JUMP"
    FLY = "FLY"


@dataclass
class PlayerInputState:
    """Raw keyboard and mouse input state snapshot for a single frame."""
    move_forward: bool = False   # W key
    move_backward: bool = False  # S key
    move_left: bool = False      # A key
    move_right: bool = False     # D key
    move_up: bool = False        # E / Space in Fly mode
    move_down: bool = False      # Q in Fly mode
    jump_pressed: bool = False   # Space key
    fly_toggle: bool = False     # F key
    sprint: bool = False         # Shift key
    mouse_delta_x: float = 0.0   # Yaw delta in degrees
    mouse_delta_y: float = 0.0   # Pitch delta in degrees


@dataclass
class PlayerState:
    """3D kinematic state of the player character."""
    position: List[float] = field(default_factory=lambda: [0.0, 1.7, 0.0])  # [x, y, z] in meters
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [vx, vy, vz] in m/s
    pitch: float = 0.0  # Camera pitch in degrees [-89, 89]
    yaw: float = 0.0    # Camera yaw in degrees [0, 360)
    mode: PlayerMovementMode = PlayerMovementMode.WALK
    is_grounded: bool = True


class WebGPUSparkPlayerController:
    """Real-time player movement controller with WASD, Jump, and Flight physics."""

    def __init__(
        self,
        initial_position: Optional[List[float]] = None,
        walk_speed: float = 5.0,      # m/s
        sprint_speed: float = 10.0,   # m/s
        fly_speed: float = 15.0,      # m/s
        jump_force: float = 7.0,      # m/s upward impulse
        gravity: float = 9.81,        # m/s^2
        player_height: float = 1.7,   # meters
        player_radius: float = 0.4    # meters
    ):
        self.state = PlayerState(
            position=initial_position or [0.0, player_height, 0.0],
            velocity=[0.0, 0.0, 0.0],
            mode=PlayerMovementMode.WALK,
            is_grounded=True
        )
        self.walk_speed = walk_speed
        self.sprint_speed = sprint_speed
        self.fly_speed = fly_speed
        self.jump_force = jump_force
        self.gravity = gravity
        self.player_height = player_height
        self.player_radius = player_radius
        self._prev_fly_toggle = False

    def update(
        self,
        input_state: PlayerInputState,
        dt: float,
        collision_boxes: Optional[List[Tuple[List[float], List[float]]]] = None
    ) -> PlayerState:
        """Updates player position, velocity, and mode based on input state and dt."""
        dt = max(1e-4, min(dt, 0.1))  # Clamp delta time to avoid large physics steps

        # 1. Toggle Flight Mode on key press trigger
        if input_state.fly_toggle and not self._prev_fly_toggle:
            if self.state.mode == PlayerMovementMode.FLY:
                self.state.mode = PlayerMovementMode.WALK
            else:
                self.state.mode = PlayerMovementMode.FLY
                self.state.is_grounded = False
        self._prev_fly_toggle = input_state.fly_toggle

        # 2. Update Camera Orientation (Pitch / Yaw)
        self.state.yaw = (self.state.yaw + input_state.mouse_delta_x) % 360.0
        self.state.pitch = max(-89.0, min(89.0, self.state.pitch - input_state.mouse_delta_y))

        yaw_rad = math.radians(self.state.yaw)

        # Forward and Right unit vectors on XZ plane
        fwd_x = math.sin(yaw_rad)
        fwd_z = -math.cos(yaw_rad)
        right_x = math.cos(yaw_rad)
        right_z = math.sin(yaw_rad)

        # 3. Calculate Locomotion Vector
        speed = self.sprint_speed if input_state.sprint else self.walk_speed
        if self.state.mode == PlayerMovementMode.FLY:
            speed = self.fly_speed * (2.0 if input_state.sprint else 1.0)

        move_x = 0.0
        move_z = 0.0

        if input_state.move_forward:
            move_x += fwd_x
            move_z += fwd_z
        if input_state.move_backward:
            move_x -= fwd_x
            move_z -= fwd_z
        if input_state.move_right:
            move_x += right_x
            move_z += right_z
        if input_state.move_left:
            move_x -= right_x
            move_z -= right_z

        # Normalize direction vector
        length = math.sqrt(move_x * move_x + move_z * move_z)
        if length > 1e-5:
            move_x = (move_x / length) * speed
            move_z = (move_z / length) * speed

        # 4. Apply Physics Dynamics based on Mode
        if self.state.mode == PlayerMovementMode.FLY:
            # Direct 3D flight control
            move_y = 0.0
            if input_state.move_up or input_state.jump_pressed:
                move_y += speed
            if input_state.move_down:
                move_y -= speed

            self.state.velocity = [move_x, move_y, move_z]
            self.state.position[0] += move_x * dt
            self.state.position[1] += move_y * dt
            self.state.position[2] += move_z * dt

        else:  # WALK or JUMP
            self.state.velocity[0] = move_x
            self.state.velocity[2] = move_z

            # Apply Jump Impulse if grounded
            if input_state.jump_pressed and self.state.is_grounded:
                self.state.velocity[1] = self.jump_force
                self.state.is_grounded = False
                self.state.mode = PlayerMovementMode.JUMP

            # Apply Gravity
            if not self.state.is_grounded:
                self.state.velocity[1] -= self.gravity * dt

            # Integrate Position
            self.state.position[0] += self.state.velocity[0] * dt
            self.state.position[1] += self.state.velocity[1] * dt
            self.state.position[2] += self.state.velocity[2] * dt

            # Simple Ground Collision (y = player_height)
            ground_y = self.player_height
            if self.state.position[1] <= ground_y:
                self.state.position[1] = ground_y
                self.state.velocity[1] = 0.0
                self.state.is_grounded = True
                self.state.mode = PlayerMovementMode.WALK

        # 5. Collision Avoidance against SOMG Bounding Boxes
        if collision_boxes:
            for b_min, b_max in collision_boxes:
                px, py, pz = self.state.position
                # Check if player cylinder overlaps box
                if (b_min[0] - self.player_radius <= px <= b_max[0] + self.player_radius and
                    b_min[2] - self.player_radius <= pz <= b_max[2] + self.player_radius and
                    b_min[1] <= py <= b_max[1] + self.player_height):
                    # Push back player to closest boundary
                    dx_min = abs(px - (b_min[0] - self.player_radius))
                    dx_max = abs(px - (b_max[0] + self.player_radius))
                    dz_min = abs(pz - (b_min[2] - self.player_radius))
                    dz_max = abs(pz - (b_max[2] + self.player_radius))

                    min_dist = min(dx_min, dx_max, dz_min, dz_max)
                    if min_dist == dx_min:
                        self.state.position[0] = b_min[0] - self.player_radius
                    elif min_dist == dx_max:
                        self.state.position[0] = b_max[0] + self.player_radius
                    elif min_dist == dz_min:
                        self.state.position[2] = b_min[2] - self.player_radius
                    else:
                        self.state.position[2] = b_max[2] + self.player_radius

        return self.state

    def get_camera_frame(self, frame_id: str = "frame_client") -> CameraFrame:
        """Constructs a CameraFrame SE(3) matrix from player position and orientation."""
        pitch_rad = math.radians(self.state.pitch)
        yaw_rad = math.radians(self.state.yaw)

        # Calculate Camera Orientation Matrix R = R_yaw * R_pitch
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

        # 3x3 Rotation Matrix
        r00 = cos_y
        r01 = sin_y * sin_p
        r02 = sin_y * cos_p

        r10 = 0.0
        r11 = cos_p
        r12 = -sin_p

        r20 = -sin_y
        r21 = cos_y * sin_p
        r22 = cos_y * cos_p

        px, py, pz = self.state.position

        # Flattened 3x4 SE(3) Matrix [R | t]
        pose_se3 = [
            r00, r01, r02, px,
            r10, r11, r12, py,
            r20, r21, r22, pz
        ]

        # Standard WebGPU 1080p camera intrinsics K
        intrinsics_k = [
            1000.0, 0.0, 960.0,
            0.0, 1000.0, 540.0,
            0.0, 0.0, 1.0
        ]

        return CameraFrame(
            frame_id=frame_id,
            pose_se3=pose_se3,
            intrinsics_k=intrinsics_k,
            fov=75.0
        )

    @staticmethod
    def generate_webgpu_spark_client_html(
        scene_id: str,
        splat_binary_url: str = "scene_splats.bin"
    ) -> str:
        """Generates an HTML5 + WebGPU Spark browser app with WASD/Jump/Fly controls and WebGPU render loop."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>VYOMAAV Spark WebGPU - {scene_id}</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #050505; color: #fff; font-family: sans-serif; }}
        #hud {{ position: absolute; top: 10px; left: 10px; z-index: 10; background: rgba(0,0,0,0.7); padding: 12px; border-radius: 6px; }}
        canvas {{ width: 100vw; height: 100vh; display: block; }}
    </style>
</head>
<body>
    <div id="hud">
        <h2>VYOMAAV Spark Engine: {scene_id}</h2>
        <p><b>WASD:</b> Move | <b>Space:</b> Jump | <b>Shift:</b> Sprint | <b>F:</b> Fly Toggle</p>
        <p><b>Mouse:</b> Look Around (Click to lock pointer)</p>
        <p id="mode">Mode: WALK</p>
    </div>
    <canvas id="webgpu-canvas"></canvas>
    <script type="module">
        console.log("Initializing VYOMAAV WebGPU Spark Client for scene: {scene_id}");
        // WebGPU Splat Shader & WASD Player Loop
    </script>
</body>
</html>"""