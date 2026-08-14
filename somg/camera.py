"""VYOMAAV Camera Trajectory & Frame Structures."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class CameraFrame:
    frame_id: str
    pose_se3: List[float]  # 12-element SE(3) matrix
    intrinsics_k: List[float]  # 9-element 3x3 K matrix
    fov: float = 75.0

class CameraTrajectoryGraph:
    """Graph container tracking camera keyframe trajectories."""

    def __init__(self):
        self.frames: Dict[str, CameraFrame] = {}

    def add_frame(self, frame: CameraFrame):
        self.frames[frame.frame_id] = frame

    def get_frame(self, frame_id: str) -> Optional[CameraFrame]:
        return self.frames.get(frame_id)
