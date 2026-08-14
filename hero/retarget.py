"""
VYOMAAV Base Model Engine
Module: hero.retarget

Hero Mode Subsystem - Sprint 13.
Tracks video protagonists, extracts parametric human body poses (SMPL-X kinematics),
and retargets motion vectors onto custom player avatars using Forward/Inverse Kinematics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math
import torch
import torch.nn as nn


@dataclass
class SMPLXBodyPose:
    """Parametric SMPL-X human body pose parameter container.

    Attributes:
        global_orient: (3,) -> Root orientation vector (axis-angle)
        transl: (3,) -> Root 3D translation in camera/world space
        body_pose: (21, 3) -> Axis-angle rotation parameters for 21 major body joints
        betas: (10,) -> Body shape distribution parameters
        expression: (10,) -> Facial expression blendshape weights
    """
    global_orient: torch.Tensor
    transl: torch.Tensor
    body_pose: torch.Tensor
    betas: torch.Tensor
    expression: torch.Tensor


@dataclass
class ActorMotionSequence:
    """Temporal sequence of extracted SMPL-X human body poses across video frames."""
    actor_id: str
    fps: float
    frames: List[SMPLXBodyPose] = field(default_factory=list)

    def num_frames(self) -> int:
        return len(self.frames)


@dataclass
class JointTransform:
    """3D Local transform matrix for an individual skeleton bone."""
    joint_name: str
    parent_index: int
    local_rotation_matrix: torch.Tensor  # (3, 3)
    local_translation: torch.Tensor       # (3,)


class SMPLXKinematicRig:
    """Standardized 22-joint SMPL-X skeletal hierarchy structure."""

    JOINT_NAMES = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_shoulder_elbow", "left_wrist", "right_wrist"
    ]

    PARENT_INDICES = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

    @classmethod
    def axis_angle_to_matrix(cls, r: torch.Tensor) -> torch.Tensor:
        """Converts 3D axis-angle rotation vector to 3x3 rotation matrix using Rodrigues' formula."""
        theta = torch.norm(r) + 1e-8
        k = r / theta
        
        K = torch.tensor([
            [0.0, -k[2].item(), k[1].item()],
            [k[2].item(), 0.0, -k[0].item()],
            [-k[1].item(), k[0].item(), 0.0]
        ], device=r.device)

        I = torch.eye(3, device=r.device)
        return I + math.sin(theta.item()) * K + (1.0 - math.cos(theta.item())) * torch.matmul(K, K)


class ParametricActorExtractor(nn.Module):
    """Neural estimator predicting SMPL-X body parameters from frame feature tokens."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # Pose prediction heads
        self.orient_head = nn.Linear(embed_dim, 3)
        self.transl_head = nn.Linear(embed_dim, 3)
        self.pose_head = nn.Linear(embed_dim, 21 * 3)
        self.beta_head = nn.Linear(embed_dim, 10)
        self.expr_head = nn.Linear(embed_dim, 10)

    def forward(self, frame_latent: torch.Tensor) -> SMPLXBodyPose:
        """Predicts SMPL-X pose parameters for a single video frame latent vector."""
        orient = self.orient_head(frame_latent).squeeze(0)
        transl = self.transl_head(frame_latent).squeeze(0)
        pose_vec = self.pose_head(frame_latent).view(21, 3)
        betas = self.beta_head(frame_latent).squeeze(0)
        expr = self.expr_head(frame_latent).squeeze(0)

        return SMPLXBodyPose(
            global_orient=orient,
            transl=transl,
            body_pose=pose_vec,
            betas=betas,
            expression=expr
        )


class MotionRetargetingEngine:
    """Retargets SMPL-X kinematic motions onto custom user avatar skeletons."""

    def __init__(self, target_avatar_height_m: float = 1.75):
        self.target_height = target_avatar_height_m

    def solve_forward_kinematics(
        self, pose: SMPLXBodyPose, device: torch.device = torch.device("cpu")
    ) -> Dict[str, torch.Tensor]:
        """Calculates global 3D joint positions (Forward Kinematics) for an SMPL-X pose."""
        global_positions: Dict[str, torch.Tensor] = {}
        global_transforms: Dict[int, torch.Tensor] = {}

        # Root Transform (Pelvis)
        root_R = SMPLXKinematicRig.axis_angle_to_matrix(pose.global_orient)
        root_T = torch.eye(4, device=device)
        root_T[:3, :3] = root_R
        root_T[:3, 3] = pose.transl
        global_transforms[0] = root_T
        global_positions[SMPLXKinematicRig.JOINT_NAMES[0]] = pose.transl

        # Traverse Joint Hierarchy
        for i in range(1, 22):
            parent_idx = SMPLXKinematicRig.PARENT_INDICES[i]
            parent_T = global_transforms[parent_idx]

            joint_axis_angle = pose.body_pose[i - 1]
            R_local = SMPLXKinematicRig.axis_angle_to_matrix(joint_axis_angle)

            # Standardized bone offset along hierarchy
            t_local = torch.tensor([0.0, 0.15, 0.0], device=device)

            T_local = torch.eye(4, device=device)
            T_local[:3, :3] = R_local
            T_local[:3, 3] = t_local

            T_global = torch.matmul(parent_T, T_local)
            global_transforms[i] = T_global
            global_positions[SMPLXKinematicRig.JOINT_NAMES[i]] = T_global[:3, 3]

        return global_positions

    def retarget_motion_to_avatar(
        self, motion_seq: ActorMotionSequence, avatar_bone_scale: float = 1.0
    ) -> List[Dict[str, torch.Tensor]]:
        """Retargets full movie actor motion sequence onto target player avatar joint channels."""
        retargeted_frames: List[Dict[str, torch.Tensor]] = []

        for frame_pose in motion_seq.frames:
            fk_positions = self.solve_forward_kinematics(frame_pose)
            
            # Apply bone scale adjustment to target avatar dimensions
            scaled_frame: Dict[str, torch.Tensor] = {}
            for joint_name, pos in fk_positions.items():
                scaled_frame[joint_name] = pos * avatar_bone_scale

            retargeted_frames.append(scaled_frame)

        return retargeted_frames