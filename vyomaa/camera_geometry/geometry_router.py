"""Geometry Router choosing appropriate geometric solvers based on input modality and evidence count."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from ..core.contracts import Camera, InputArtifact, Observation
from ..core.exceptions import CameraGeometryError
from ..core.types import ModalityType


class GeometryRouter:
    """Determines the optimal geometric solver strategy matching input modality constraints."""

    @staticmethod
    def select_solver(input_artifact: InputArtifact) -> str:
        """Route to appropriate solver strategy without fabricating world poses for single images."""
        if input_artifact.modality == ModalityType.RGB_IMAGE:
            # Single image -> strictly image-space perspective, no synthetic world trajectories
            return "single_image_perspective"
        elif input_artifact.modality == ModalityType.MULTIVIEW_IMAGE_SET:
            # Multi-view set -> feature correspondence + epipolar relative poses + bundle adjustment
            return "multiview_correspondence_sfm"
        elif input_artifact.modality in {ModalityType.MONOCULAR_VIDEO, ModalityType.MULTI_CAMERA_VIDEO}:
            # Video sequence -> keyframe tracking + trajectory smoothing + bundle adjustment
            return "video_temporal_sfm"
        elif input_artifact.modality in {ModalityType.RGBD_IMAGE, ModalityType.RGBD_VIDEO}:
            # Direct sensor depth -> unprojection + metric scale
            return "rgbd_metric_direct"
        else:
            return "single_image_perspective"
