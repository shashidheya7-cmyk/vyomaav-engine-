echo "=================================================="
echo " 🏗️ SETTING UP VYOMAAV PHASE 4B.1 FOUNDATION"
echo "==================================================\n"

# 1. Create Directories
mkdir -p vyomaa/multiview
mkdir -p vyomaa/camera_geometry
mkdir -p vyomaa/validation
mkdir -p vyomaa/fusion
mkdir -p vyomaa/pipeline
mkdir -p tests
mkdir -p reports/phase4b

# Create __init__.py files
touch vyomaa/__init__.py
touch vyomaa/multiview/__init__.py
touch vyomaa/camera_geometry/__init__.py
touch vyomaa/validation/__init__.py
touch vyomaa/fusion/__init__.py
touch vyomaa/pipeline/__init__.py

echo "[✓] Directory structure initialized."

# 2. Write vyomaa/multiview/contracts.py
cat << 'EOT' > vyomaa/multiview/contracts.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

@dataclass
class ViewSet:
    observation_ids: List[str]
    timestamps: List[float] = field(default_factory=list)
    keyframe_flags: List[bool] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    image_quality_scores: List[float] = field(default_factory=list)
    selected_view_confidence: List[float] = field(default_factory=list)
    source_modality: str = "rgb"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_ids": self.observation_ids,
            "timestamps": self.timestamps,
            "keyframe_flags": self.keyframe_flags,
            "image_paths": self.image_paths,
            "image_quality_scores": self.image_quality_scores,
            "selected_view_confidence": self.selected_view_confidence,
            "source_modality": self.source_modality,
            "provenance": self.provenance
        }

@dataclass
class CameraEstimate:
    camera_id: str
    intrinsics_k: np.ndarray
    extrinsics_rt: np.ndarray
    focal_lengths: tuple[float, float]
    principal_point: tuple[float, float]
    backend_name: str
    coordinate_convention: str = "opencv"
    confidence: float = 1.0
    reprojection_error: float = 0.0
    validity_state: bool = True
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "intrinsics_k": self.intrinsics_k.tolist() if isinstance(self.intrinsics_k, np.ndarray) else self.intrinsics_k,
            "extrinsics_rt": self.extrinsics_rt.tolist() if isinstance(self.extrinsics_rt, np.ndarray) else self.extrinsics_rt,
            "focal_lengths": self.focal_lengths,
            "principal_point": self.principal_point,
            "backend_name": self.backend_name,
            "coordinate_convention": self.coordinate_convention,
            "confidence": self.confidence,
            "reprojection_error": self.reprojection_error,
            "validity_state": self.validity_state,
            "provenance": self.provenance
        }

@dataclass
class DenseGeometry:
    source_observation_id: str
    depth_array_shape: tuple[int, ...]
    point_map_shape: tuple[int, ...]
    validity_mask_shape: tuple[int, ...]
    confidence: float
    resolution: tuple[int, int]
    backend: str
    coordinate_space: str = "camera"
    provenance: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorrespondenceSet:
    source_observation_id: str
    target_observation_id: str
    correspondences_2d: np.ndarray
    confidence: float
    inlier_count: int
    inlier_ratio: float
    geometric_model: str = "fundamental"
    provenance: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeometryEvidence:
    backend: str
    cameras: List[CameraEstimate]
    dense_geometry: List[DenseGeometry]
    correspondences: List[CorrespondenceSet]
    confidence: float
    reprojection_metrics: Dict[str, float]
    consistency_metrics: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FusedWorldGeometry:
    fused_points: np.ndarray
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    source_view_ids: List[str] = field(default_factory=list)
    object_ids: List[str] = field(default_factory=list)
    coordinate_frame: str = "world"
    bounds: Dict[str, list[float]] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
EOT
echo "[✓] Created contracts.py"

# 3. Write vyomaa/multiview/keyframe_selector.py
cat << 'EOT' > vyomaa/multiview/keyframe_selector.py
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from vyomaa.multiview.contracts import ViewSet

logger = logging.getLogger("vyomaa.multiview.keyframe_selector")

class KeyframeSelector:
    def __init__(
        self,
        max_frames: int = 100,
        temporal_window: int = 5,
        sharpness_threshold: float = 100.0,
        similarity_threshold: float = 0.85
    ):
        self.max_frames = max_frames
        self.temporal_window = temporal_window
        self.sharpness_threshold = sharpness_threshold
        self.similarity_threshold = similarity_threshold

    def select_keyframes(
        self,
        observation_ids: List[str],
        timestamps: List[float],
        image_paths: List[str],
        quality_scores: Optional[List[float]] = None
    ) -> ViewSet:
        n = len(observation_ids)
        if n == 0:
            return ViewSet(observation_ids=[], timestamps=[], keyframe_flags=[], image_paths=[])

        if quality_scores is None:
            quality_scores = [1.0] * n

        keyframe_flags = [False] * n
        keyframe_flags[0] = True
        last_selected_idx = 0

        selected_count = 1
        for i in range(1, n):
            time_diff = timestamps[i] - timestamps[last_selected_idx] if timestamps else float(i)
            quality_ok = quality_scores[i] >= (self.sharpness_threshold * 0.01)

            if (i - last_selected_idx >= self.temporal_window) or (quality_ok and time_diff > 1.0):
                keyframe_flags[i] = True
                last_selected_idx = i
                selected_count += 1
                if selected_count >= self.max_frames:
                    break

        if n > 1 and not keyframe_flags[-1]:
            keyframe_flags[-1] = True

        selected_ids = [obs for obs, flag in zip(observation_ids, keyframe_flags) if flag]
        selected_ts = [ts for ts, flag in zip(timestamps, keyframe_flags) if flag] if timestamps else []
        selected_paths = [p for p, flag in zip(image_paths, keyframe_flags) if flag]
        selected_scores = [q for q, flag in zip(quality_scores, keyframe_flags) if flag]

        provenance = {
            "selector": "KeyframeSelector",
            "total_input_frames": n,
            "selected_keyframes": len(selected_ids),
            "temporal_window": self.temporal_window
        }

        return ViewSet(
            observation_ids=selected_ids,
            timestamps=selected_ts,
            keyframe_flags=[True] * len(selected_ids),
            image_paths=selected_paths,
            image_quality_scores=selected_scores,
            selected_view_confidence=selected_scores,
            source_modality="rgb",
            provenance=provenance
        )
EOT
echo "[✓] Created keyframe_selector.py"

# 4. Write vyomaa/multiview/view_graph.py
cat << 'EOT' > vyomaa/multiview/view_graph.py
import logging
from typing import Dict, List, Any
from vyomaa.multiview.contracts import ViewSet

logger = logging.getLogger("vyomaa.multiview.view_graph")

class ViewGraphNode:
    def __init__(self, observation_id: str, index: int, timestamp: float, metadata: Dict[str, Any]):
        self.observation_id = observation_id
        self.index = index
        self.timestamp = timestamp
        self.metadata = metadata

class ViewGraphEdge:
    def __init__(self, source_id: str, target_id: str, edge_type: str, weight: float, confidence: float, attributes: Dict[str, Any]):
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type
        self.weight = weight
        self.confidence = confidence
        self.attributes = attributes

class ViewGraph:
    def __init__(self):
        self.nodes: Dict[str, ViewGraphNode] = {}
        self.edges: List[ViewGraphEdge] = []

    def add_node(self, observation_id: str, index: int, timestamp: float, metadata: Optional[Dict[str, Any]] = None):
        self.nodes[observation_id] = ViewGraphNode(observation_id, index, timestamp, metadata or {})

    def add_edge(self, source_id: str, target_id: str, edge_type: str, weight: float, confidence: float, attributes: Optional[Dict[str, Any]] = None):
        self.edges.append(ViewGraphEdge(source_id, target_id, edge_type, weight, confidence, attributes or {}))

    @classmethod
    def from_view_set(cls, view_set: ViewSet, temporal_window: int = 2) -> "ViewGraph":
        graph = cls()
        ids = view_set.observation_ids
        timestamps = view_set.timestamps if view_set.timestamps else [float(i) for i in range(len(ids))]

        for i, obs_id in enumerate(ids):
            graph.add_node(obs_id, i, timestamps[i], {"image_path": view_set.image_paths[i] if i < len(view_set.image_paths) else ""})

        n = len(ids)
        for i in range(n):
            for j in range(i + 1, min(i + 1 + temporal_window, n)):
                src, tgt = ids[i], ids[j]
                time_delta = abs(timestamps[j] - timestamps[i])
                weight = 1.0 / (1.0 + time_delta)
                graph.add_edge(src, tgt, "temporal_adjacency", weight=weight, confidence=0.95, attributes={"time_delta": time_delta})

        return graph

    def get_local_neighbors(self, observation_id: str, k: int = 3) -> List[str]:
        neighbors = []
        for edge in self.edges:
            if edge.source_id == observation_id and edge.edge_type == "temporal_adjacency":
                neighbors.append(edge.target_id)
        return neighbors[:k]
EOT
echo "[✓] Created view_graph.py"

# 5. Write vyomaa/camera_geometry/base.py
cat << 'EOT' > vyomaa/camera_geometry/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from vyomaa.multiview.contracts import ViewSet, GeometryEvidence

class BaseGeometryBackend(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = getattr(torch, config.get("dtype", "float32"))
        self.batch_limit = config.get("batch_limit", 4)
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def estimate_geometry(self, view_set: ViewSet) -> GeometryEvidence:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        pass
EOT
echo "[✓] Created camera_geometry/base.py"

# 6. Write vyomaa/camera_geometry/geometry_router.py
cat << 'EOT' > vyomaa/camera_geometry/geometry_router.py
import logging
from typing import Dict, Any, Optional
from vyomaa.multiview.contracts import ViewSet, GeometryEvidence
from vyomaa.camera_geometry.base import BaseGeometryBackend

logger = logging.getLogger("vyomaa.camera_geometry.geometry_router")

class GeometryRouter:
    def __init__(self, backends: Dict[str, BaseGeometryBackend], policy_config: Optional[Dict[str, Any]] = None):
        self.backends = backends
        self.policy_config = policy_config or {
            "primary": "vggt",
            "fallback": "dust3r",
            "specialized": "mast3r",
            "classical_fallback": "colmap",
            "confidence_threshold": 0.75,
            "enable_disagreement_logging": True
        }

    def route(self, view_set: ViewSet) -> GeometryEvidence:
        primary_name = self.policy_config.get("primary", "vggt")
        fallback_name = self.policy_config.get("fallback", "dust3r")

        backend = self.backends.get(primary_name)
        if backend and backend.is_available():
            evidence = backend.estimate_geometry(view_set)
            if evidence.confidence >= self.policy_config.get("confidence_threshold", 0.75):
                evidence.provenance["routing_decision"] = f"primary_{primary_name}"
                return evidence

        fallback_backend = self.backends.get(fallback_name)
        if fallback_backend and fallback_backend.is_available():
            evidence = fallback_backend.estimate_geometry(view_set)
            evidence.provenance["routing_decision"] = f"fallback_{fallback_name}"
            return evidence

        classical_name = self.policy_config.get("classical_fallback", "colmap")
        classical_backend = self.backends.get(classical_name)
        if classical_backend and classical_backend.is_available():
            evidence = classical_backend.estimate_geometry(view_set)
            evidence.provenance["routing_decision"] = f"classical_fallback_{classical_name}"
            return evidence

        raise RuntimeError("No available geometry backends could successfully process the ViewSet.")
EOT
echo "[✓] Created geometry_router.py"

# 7. Write vyomaa/validation/camera_validation.py
cat << 'EOT' > vyomaa/validation/camera_validation.py
import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger("vyomaa.validation.camera_validation")

class CameraValidator:
    @staticmethod
    def validate_camera_matrix(k: np.ndarray, rt: np.ndarray, image_size: tuple[int, int]) -> Dict[str, Any]:
        errors = []

        if not np.isfinite(k).all() or not np.isfinite(rt).all():
            raise ValueError("Camera intrinsics or extrinsics contain NaN or Inf values.")

        if k.shape != (3, 3):
            errors.append(f"Invalid intrinsics shape {k.shape}, expected (3, 3).")

        fx, fy = k[0, 0], k[1, 1]
        if fx <= 0 or fy <= 0:
            errors.append(f"Non-positive focal lengths detected: fx={fx}, fy={fy}.")

        cx, cy = k[0, 2], k[1, 2]
        width, height = image_size
        if not (0 <= cx <= width and 0 <= cy <= height):
            errors.append(f"Principal point ({cx}, {cy}) outside image dimensions ({width}x{height}).")

        r = rt[:3, :3]
        det = np.linalg.det(r)
        if abs(det - 1.0) > 1e-2:
            errors.append(f"Rotation matrix determinant is {det:.4f}, expected approx +1.0.")

        ortho_error = np.linalg.norm(r @ r.T - np.eye(3))
        if ortho_error > 1e-2:
            errors.append(f"Rotation matrix is not orthogonal (ortho error: {ortho_error:.4f}).")

        if errors:
            return {"valid": False, "errors": errors}

        return {"valid": True, "errors": []}
EOT
echo "[✓] Created camera_validation.py"

# 8. Write Fusion Modules
cat << 'EOT' > vyomaa/fusion/outlier_rejection.py
import numpy as np

class OutlierRejection:
    @staticmethod
    def statistical_outlier_removal(points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
        if len(points) == 0:
            return points
        from scipy.spatial import KDTree
        tree = KDTree(points)
        distances, _ = tree.query(points, k=min(nb_neighbors + 1, len(points)))
        mean_distances = np.mean(distances[:, 1:], axis=1) if distances.shape[1] > 1 else np.zeros(len(points))
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        threshold = global_mean + std_ratio * global_std
        mask = mean_distances < threshold
        return points[mask]
EOT

cat << 'EOT' > vyomaa/fusion/normal_fusion.py
import numpy as np

class NormalFusion:
    @staticmethod
    def compute_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
        if len(points) < 3:
            return np.zeros_like(points)
        from scipy.spatial import KDTree
        tree = KDTree(points)
        normals = np.zeros_like(points)
        for i, p in enumerate(points):
            _, idx = tree.query(p, k=min(k, len(points)))
            neighbors = points[idx]
            cov = np.cov(neighbors.T)
            _, eigenvecs = np.linalg.eigh(cov)
            normals[i] = eigenvecs[:, 0]
        return normals
EOT

cat << 'EOT' > vyomaa/fusion/dynamic_mask.py
from typing import List, Optional

class DynamicMaskInterface:
    def __init__(self, excluded_object_ids: Optional[List[str]] = None, excluded_classes: Optional[List[str]] = None):
        self.excluded_object_ids = excluded_object_ids or []
        self.excluded_classes = excluded_classes or ["dynamic_person", "moving_vehicle"]

    def should_exclude(self, object_id: Optional[str] = None, object_class: Optional[str] = None) -> bool:
        if object_id and object_id in self.excluded_object_ids:
            return True
        if object_class and object_class in self.excluded_classes:
            return True
        return False
EOT

cat << 'EOT' > vyomaa/fusion/dense_point_fusion.py
import logging
import numpy as np
from typing import List, Optional
from vyomaa.multiview.contracts import CameraEstimate, FusedWorldGeometry
from vyomaa.fusion.outlier_rejection import OutlierRejection
from vyomaa.fusion.normal_fusion import NormalFusion

logger = logging.getLogger("vyomaa.fusion.dense_point_fusion")

class DensePointFusion:
    def __init__(self, chunk_size: int = 100000, std_ratio: float = 2.0):
        self.chunk_size = chunk_size
        self.std_ratio = std_ratio

    def fuse(
        self,
        cameras: List[CameraEstimate],
        depth_maps: List[np.ndarray],
        confidences: Optional[List[np.ndarray]] = None,
        colors: Optional[List[np.ndarray]] = None
    ) -> FusedWorldGeometry:
        all_points = []
        all_colors = []
        all_conf = []

        for idx, (cam, depth) in enumerate(zip(cameras, depth_maps)):
            h, w = depth.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            u = u.flatten()
            v = v.flatten()
            z = depth.flatten()

            valid = z > 0
            if not valid.any():
                continue

            u = u[valid]
            v = v[valid]
            z = z[valid]

            fx, fy = cam.focal_lengths
            cx, cy = cam.principal_point

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)

            rt = cam.extrinsics_rt
            if rt.shape == (3, 4):
                rt_full = np.eye(4)
                rt_full[:3, :4] = rt
                rt = rt_full

            r = rt[:3, :3]
            t = rt[:3, 3]
            points_world = (r @ points_cam[:, :3].T).T + t

            all_points.append(points_world)

            if colors is not None and idx < len(colors):
                c_flat = colors[idx].reshape(-1, 3)[valid]
                all_colors.append(c_flat)

            if confidences is not None and idx < len(confidences):
                conf_flat = confidences[idx].flatten()[valid]
                all_conf.append(conf_flat)

        if not all_points:
            return FusedWorldGeometry(fused_points=np.zeros((0, 3)))

        fused_points = np.concatenate(all_points, axis=0)
        fused_colors = np.concatenate(all_colors, axis=0) if all_colors else None
        fused_conf = np.concatenate(all_conf, axis=0) if all_conf else None

        fused_points = OutlierRejection.statistical_outlier_removal(fused_points, std_ratio=self.std_ratio)
        normals = NormalFusion.compute_normals(fused_points)

        bounds = {
            "min": fused_points.min(axis=0).tolist() if len(fused_points) > 0 else [0, 0, 0],
            "max": fused_points.max(axis=0).tolist() if len(fused_points) > 0 else [0, 0, 0]
        }

        provenance = {
            "fusion_engine": "DensePointFusion",
            "total_fused_points": len(fused_points),
            "input_views": len(cameras)
        }

        return FusedWorldGeometry(
            fused_points=fused_points,
            normals=normals,
            colors=fused_colors,
            confidence=fused_conf,
            bounds=bounds,
            provenance=provenance
        )
EOT
echo "[✓] Created fusion modules."

# 9. Write Unit Tests: tests/test_multiview_foundation.py
cat << 'EOT' > tests/test_multiview_foundation.py
import unittest
import numpy as np
from vyomaa.multiview.contracts import ViewSet, CameraEstimate, FusedWorldGeometry
from vyomaa.multiview.keyframe_selector import KeyframeSelector
from vyomaa.multiview.view_graph import ViewGraph
from vyomaa.validation.camera_validation import CameraValidator
from vyomaa.fusion.dense_point_fusion import DensePointFusion
from vyomaa.fusion.outlier_rejection import OutlierRejection

class TestMultiViewFoundation(unittest.TestCase):
    def test_view_set_serialization(self):
        vs = ViewSet(
            observation_ids=["obs_1", "obs_2"],
            timestamps=[0.0, 1.0],
            keyframe_flags=[True, True],
            image_paths=["/tmp/1.jpg", "/tmp/2.jpg"]
        )
        d = vs.to_dict()
        self.assertEqual(d["observation_ids"], ["obs_1", "obs_2"])
        self.assertTrue(d["keyframe_flags"][0])

    def test_keyframe_selector(self):
        selector = KeyframeSelector(temporal_window=2)
        obs_ids = [f"obs_{i}" for i in range(10)]
        ts = [float(i) * 0.1 for i in range(10)]
        paths = [f"/tmp/{i}.jpg" for i in range(10)]
        
        vs = selector.select_keyframes(obs_ids, ts, paths)
        self.assertGreater(len(vs.observation_ids), 0)
        self.assertLessEqual(len(vs.observation_ids), 10)

    def test_view_graph_construction(self):
        vs = ViewSet(
            observation_ids=["obs_1", "obs_2", "obs_3"],
            timestamps=[0.0, 1.0, 2.0],
            keyframe_flags=[True, True, True],
            image_paths=["a.jpg", "b.jpg", "c.jpg"]
        )
        graph = ViewGraph.from_view_set(vs, temporal_window=1)
        self.assertEqual(len(graph.nodes), 3)
        self.assertGreater(len(graph.edges), 0)
        neighbors = graph.get_local_neighbors("obs_1")
        self.assertIn("obs_2", neighbors)

    def test_camera_validation(self):
        k = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        rt = np.eye(4)[:3, :]
        res = CameraValidator.validate_camera_matrix(k, rt, (640, 480))
        self.assertTrue(res["valid"])

        rt_bad = rt.copy()
        rt_bad[0, 0] = -5.0
        res_bad = CameraValidator.validate_camera_matrix(k, rt_bad, (640, 480))
        self.assertFalse(res_bad["valid"])

    def test_dense_point_fusion_and_outliers(self):
        points = np.random.rand(100, 3)
        outliers = np.array([[100.0, 100.0, 100.0]])
        noisy_points = np.concatenate([points, outliers], axis=0)

        filtered = OutlierRejection.statistical_outlier_removal(noisy_points, nb_neighbors=5, std_ratio=1.5)
        self.assertLess(len(filtered), len(noisy_points))

    def test_fusion_orchestration(self):
        cam = CameraEstimate(
            camera_id="cam_1",
            intrinsics_k=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
            extrinsics_rt=np.eye(4)[:3, :],
            focal_lengths=(500.0, 500.0),
            principal_point=(320.0, 240.0),
            backend_name="test"
        )
        depth = np.ones((10, 10), dtype=np.float32) * 2.0
        fusion = DensePointFusion()
        fused = fusion.fuse([cam], [depth])
        self.assertIsInstance(fused, FusedWorldGeometry)

if __name__ == "__main__":
    unittest.main()
EOT
echo "[✓] Created tests/test_multiview_foundation.py"

echo "\n=================================================="
echo " 🧪 RUNNING TEST SUITE..."
echo "=================================================="
/root/miniconda3/envs/py3.10/bin/python3 -m unittest tests/test_multiview_foundation.py
