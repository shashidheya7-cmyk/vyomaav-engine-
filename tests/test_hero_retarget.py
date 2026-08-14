"""
VYOMAAV Base Model Engine
Test Suite: tests/test_hero_retarget.py

Pytest suite validating Sprint 13: Parametric SMPL-X pose prediction, Rodrigues axis-angle matrix conversion,
Forward Kinematics joint hierarchy solving, and avatar motion retargeting.
"""

import pytest
import torch
from hero.retarget import (
    SMPLXBodyPose, ActorMotionSequence, SMPLXKinematicRig,
    ParametricActorExtractor, MotionRetargetingEngine
)


def test_rodrigues_axis_angle_conversion():
    axis_angle = torch.tensor([0.0, 0.0, 0.0])  # Zero rotation
    R = SMPLXKinematicRig.axis_angle_to_matrix(axis_angle)

    assert R.shape == (3, 3)
    assert torch.allclose(R, torch.eye(3), atol=1e-4)


def test_parametric_actor_extractor_prediction():
    extractor = ParametricActorExtractor(embed_dim=64)
    frame_latent = torch.randn(1, 64)

    pose = extractor(frame_latent)

    assert isinstance(pose, SMPLXBodyPose)
    assert pose.global_orient.shape == (3,)
    assert pose.transl.shape == (3,)
    assert pose.body_pose.shape == (21, 3)
    assert pose.betas.shape == (10,)
    assert pose.expression.shape == (10,)


def test_forward_kinematics_and_motion_retargeting():
    engine = MotionRetargetingEngine(target_avatar_height_m=1.80)

    # Sample pose
    pose = SMPLXBodyPose(
        global_orient=torch.tensor([0.0, 0.0, 0.0]),
        transl=torch.tensor([0.0, 1.0, 0.0]),  # 1m above ground
        body_pose=torch.zeros((21, 3)),
        betas=torch.zeros(10),
        expression=torch.zeros(10)
    )

    fk_positions = engine.solve_forward_kinematics(pose)

    assert "pelvis" in fk_positions
    assert "head" in fk_positions
    assert torch.allclose(fk_positions["pelvis"], torch.tensor([0.0, 1.0, 0.0]))

    # Retarget Sequence
    seq = ActorMotionSequence(actor_id="hero_actor_01", fps=30.0, frames=[pose, pose])
    retargeted = engine.retarget_motion_to_avatar(seq, avatar_bone_scale=1.1)

    assert len(retargeted) == 2
    assert "head" in retargeted[0]