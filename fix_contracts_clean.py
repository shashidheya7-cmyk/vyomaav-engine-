from pathlib import Path
import subprocess
import sys

contracts_code = '''"""
Multi-View Contracts & Representations for VYOMAAV Engine
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Re-export canonical FusedWorldGeometry & DynamicCluster
from vyomaa.representations.fused_world import FusedWorldGeometry, DynamicCluster

@dataclass
class ViewSet:
    observation_ids: List[str]
    timestamps: List[float] = field(default_factory=list)
    keyframe_flags: List[bool] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_ids": self.observation_ids,
            "timestamps": self.timestamps,
            "keyframe_flags": self.keyframe_flags,
            "image_paths": self.image_paths,
            "metadata": self.metadata
        }

@dataclass
class CameraEstimate:
    camera_id: str
    intrinsics_k: np.ndarray
    extrinsics_rt: np.ndarray
    focal_lengths: Tuple[float, float] = (500.0, 500.0)
    principal_point: Tuple[float, float] = (256.0, 256.0)
    backend_name: str = "vggt"
    coordinate_convention: str = "opencv"
    confidence: float = 1.0
    reprojection_error: float = 0.0
    validity_state: bool = True
    provenance: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DenseGeometry:
    source_observation_id: str
    depth_array_shape: Tuple[int, int]
    point_map_shape: Tuple[int, int, int]
    validity_mask_shape: Tuple[int, int]
    confidence: float = 1.0
    resolution: Tuple[int, int] = (512, 512)
    backend: str = "vggt"
    coordinate_space: str = "camera"
    provenance: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorrespondenceSet:
    source_observation_id: str
    target_observation_id: str
    correspondences_2d: np.ndarray
    confidence: float = 1.0
    inlier_count: int = 0
    inlier_ratio: float = 1.0
    geometric_model: str = "fundamental"
    provenance: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeometryEvidence:
    backend: str
    cameras: List[CameraEstimate] = field(default_factory=list)
    dense_geometry: List[DenseGeometry] = field(default_factory=list)
    correspondences: List[CorrespondenceSet] = field(default_factory=list)
    confidence: float = 1.0
    reprojection_metrics: Dict[str, Any] = field(default_factory=dict)
    consistency_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
'''

with open("vyomaa/multiview/contracts.py", "w") as f:
    f.write(contracts_code)

print("[✓] Cleanly rewrote vyomaa/multiview/contracts.py with canonical FusedWorldGeometry import.")

# Run full test suite
cmd = [
    sys.executable, "-m", "unittest",
    "tests/test_phase4c_fusion.py",
    "tests/test_vggt_real_gpu.py",
    "tests/test_sam2_multiframe.py",
    "tests/test_sam2_real_gpu.py",
    "tests/test_sam2_worker.py",
    "tests/test_multiview_foundation.py"
]
res = subprocess.run(cmd)
sys.exit(res.returncode)
