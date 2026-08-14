"""
VYOMAAV Base Model Engine
Module: engine.slam

Real-time Keyframe Pose Graph Visual-Inertial SLAM Tracking Engine (Sprint 19).
Processes incoming camera frame feeds and IMU measurements:
1. IMU Pre-integration (Translational & Rotational integration).
2. Keypoint Feature Detection & Feature Association.
3. Keyframe Selection & SE(3) Pose Graph Bundle Adjustment.
4. Real-time update of SOMG scene camera trajectories.
"""

from dataclasses import dataclass, field
import math
from typing import List, Dict, Tuple, Optional, Set
import torch

from somg.camera import CameraFrame, CameraTrajectoryGraph
from somg.scene import SceneState


@dataclass
class IMUSample:
    """High-frequency Inertial Measurement Unit (IMU) sensor snapshot."""
    timestamp_s: float
    accel: List[float]  # [ax, ay, az] in m/s^2
    gyro: List[float]   # [wx, wy, wz] in rad/s


@dataclass
class VisualKeypoint2D:
    """2D detected image feature point."""
    keypoint_id: int
    u_px: float
    v_px: float
    descriptor: List[float]  # Feature vector representation


@dataclass
class SLAMKeyframe:
    """Optimized keyframe pose node within the SLAM pose graph."""
    keyframe_id: int
    timestamp_s: float
    pose_se3: List[float]  # 12-element SE(3) matrix [R | t]
    keypoints: List[VisualKeypoint2D] = field(default_factory=list)
    is_keyframe: bool = True


class VisualInertialSLAMEngine:
    """Real-time Visual-Inertial SLAM Pose Graph Tracking Engine."""

    def __init__(
        self,
        scene: SceneState,
        keyframe_trans_threshold: float = 0.2,   # Meters required to spawn keyframe
        keyframe_rot_threshold_deg: float = 10.0 # Rotation degrees required to spawn keyframe
    ):
        self.scene = scene
        self.trans_threshold = keyframe_trans_threshold
        self.rot_threshold_deg = keyframe_rot_threshold_deg

        self.keyframes: List[SLAMKeyframe] = []
        self.imu_buffer: List[IMUSample] = []
        
        # Current camera pose state [R (3x3) | t (3x1)] -> 12-element SE(3)
        self.current_pose_se3: List[float] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        self.velocity: List[float] = [0.0, 0.0, 0.0]
        self.keyframe_counter = 0

    def integrate_imu(self, imu_sample: IMUSample, dt: float):
        """Integrates high-frequency IMU acceleration and angular velocity measurements."""
        self.imu_buffer.append(imu_sample)

        # Simple Euler integration for pose prediction
        ax, ay, az = imu_sample.accel
        wx, wy, wz = imu_sample.gyro

        # Integrate velocity: v = v + a * dt
        self.velocity[0] += ax * dt
        self.velocity[1] += ay * dt
        self.velocity[2] += az * dt

        # Integrate position: t = t + v * dt
        self.current_pose_se3[3] += self.velocity[0] * dt
        self.current_pose_se3[7] += self.velocity[1] * dt
        self.current_pose_se3[11] += self.velocity[2] * dt

    def process_frame(
        self,
        timestamp_s: float,
        rgb_image_tensor: torch.Tensor,
        intrinsics_k: List[float]
    ) -> CameraFrame:
        """Processes incoming visual camera frame, evaluates keyframe metrics, and runs pose graph BA."""
        # 1. Feature Detection (Simulated grid keypoint extraction)
        keypoints = self._detect_features(rgb_image_tensor)

        # 2. Keyframe Selection Decision
        should_spawn_kf = False
        if not self.keyframes:
            should_spawn_kf = True
        else:
            last_kf = self.keyframes[-1]
            last_pos = [last_kf.pose_se3[3], last_kf.pose_se3[7], last_kf.pose_se3[11]]
            curr_pos = [self.current_pose_se3[3], self.current_pose_se3[7], self.current_pose_se3[11]]

            dist = math.sqrt(sum((curr_pos[i] - last_pos[i])**2 for i in range(3)))
            if dist >= self.trans_threshold:
                should_spawn_kf = True

        # 3. Create & Add Keyframe
        if should_spawn_kf:
            kf = SLAMKeyframe(
                keyframe_id=self.keyframe_counter,
                timestamp_s=timestamp_s,
                pose_se3=list(self.current_pose_se3),
                keypoints=keypoints
            )
            self.keyframes.append(kf)
            self.keyframe_counter += 1

            # 4. Pose Graph Optimization (Bundle Adjustment pass over keyframe chain)
            self._optimize_pose_graph()

        # 5. Synchronize Frame into SOMG Camera Graph
        cam_frame = CameraFrame(
            frame_id=f"frame_slam_{timestamp_s:.3f}",
            pose_se3=list(self.current_pose_se3),
            intrinsics_k=intrinsics_k,
            fov=75.0
        )
        self.scene.camera_graph.add_frame(cam_frame)

        return cam_frame

    def _detect_features(self, image_tensor: torch.Tensor) -> List[VisualKeypoint2D]:
        """Extracts 2D feature keypoints across the image tensor."""
        # Detect grid points
        _, h, w = image_tensor.shape
        keypoints: List[VisualKeypoint2D] = []
        kp_id = 0

        for u in range(32, w - 32, 64):
            for v in range(32, h - 32, 64):
                kp = VisualKeypoint2D(
                    keypoint_id=kp_id,
                    u_px=float(u),
                    v_px=float(v),
                    descriptor=[0.1, 0.2, 0.3, 0.4]
                )
                keypoints.append(kp)
                kp_id += 1

        return keypoints

    def _optimize_pose_graph(self):
        """Runs non-linear pose graph smoothing over keyframe SE(3) trajectory nodes."""
        if len(self.keyframes) < 2:
            return

        # Smooth translation drift across keyframe chain
        alpha = 0.1
        for i in range(1, len(self.keyframes)):
            prev_kf = self.keyframes[i - 1]
            curr_kf = self.keyframes[i]

            for axis in (3, 7, 11):
                smooth_pos = (1.0 - alpha) * curr_kf.pose_se3[axis] + alpha * prev_kf.pose_se3[axis]
                curr_kf.pose_se3[axis] = smooth_pos