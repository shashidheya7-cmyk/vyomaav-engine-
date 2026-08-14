"""Canonical typed data models for all artifacts and observations in the VYOMAA Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base_artifact import BaseArtifact
from .exceptions import SchemaValidationError
from .metadata import ArtifactMetadata
from .provenance import ProvenanceRecord
from .types import ArtifactType, ConfidenceLevel, EntityType, ModalityType


@dataclass
class InputArtifact(BaseArtifact):
    """Raw ingestion artifact entering the engine pipeline."""

    modality: ModalityType = ModalityType.RGB_IMAGE
    file_path: Optional[str] = None
    file_paths: List[str] = field(default_factory=list)
    resolution: Tuple[int, int] = (0, 0)  # (width, height)
    channels: int = 3
    fps: Optional[float] = None
    duration_seconds: Optional[float] = None
    color_space: str = "sRGB"

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.INPUT_MEDIA
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "modality": self.modality.value if isinstance(self.modality, ModalityType) else str(self.modality),
            "file_path": self.file_path,
            "file_paths": self.file_paths,
            "resolution": list(self.resolution),
            "channels": self.channels,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "color_space": self.color_space,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InputArtifact:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType(kwargs.get("artifact_type", ArtifactType.INPUT_MEDIA))
        if "modality" in kwargs and isinstance(kwargs["modality"], str):
            kwargs["modality"] = ModalityType(kwargs["modality"])
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        if "confidence_level" in kwargs and isinstance(kwargs["confidence_level"], str):
            kwargs["confidence_level"] = ConfidenceLevel(kwargs["confidence_level"])
        if "resolution" in kwargs and isinstance(kwargs["resolution"], list):
            kwargs["resolution"] = tuple(kwargs["resolution"])
        return cls(**kwargs)


@dataclass
class FrameArtifact(BaseArtifact):
    """Single decoded frame from an image set or video sequence."""

    frame_index: int = 0
    timestamp_seconds: float = 0.0
    sequence_id: Optional[str] = None
    image_path: Optional[str] = None
    resolution: Tuple[int, int] = (0, 0)
    is_keyframe: bool = False
    quality_score: float = 1.0

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.FRAME
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "sequence_id": self.sequence_id,
            "image_path": self.image_path,
            "resolution": list(self.resolution),
            "is_keyframe": self.is_keyframe,
            "quality_score": self.quality_score,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FrameArtifact:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.FRAME
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        if "confidence_level" in kwargs and isinstance(kwargs["confidence_level"], str):
            kwargs["confidence_level"] = ConfidenceLevel(kwargs["confidence_level"])
        if "resolution" in kwargs and isinstance(kwargs["resolution"], list):
            kwargs["resolution"] = tuple(kwargs["resolution"])
        return cls(**kwargs)


@dataclass
class MediaSequence(BaseArtifact):
    """Ordered collection of video frames or multi-camera streams."""

    frames: List[FrameArtifact] = field(default_factory=list)
    total_frames: int = 0
    fps: float = 30.0
    camera_model: Optional[str] = None

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.MEDIA_SEQUENCE
        if self.frames and not self.total_frames:
            self.total_frames = len(self.frames)
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "frames": [f.to_dict() for f in self.frames],
            "total_frames": self.total_frames,
            "fps": self.fps,
            "camera_model": self.camera_model,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MediaSequence:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.MEDIA_SEQUENCE
        if "frames" in kwargs:
            kwargs["frames"] = [FrameArtifact.from_dict(f) for f in kwargs["frames"]]
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        return cls(**kwargs)


@dataclass
class Camera(BaseArtifact):
    """Pinhole camera representation with intrinsics and extrinsics."""

    intrinsic_matrix: List[List[float]] = field(
        default_factory=lambda: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    extrinsic_matrix: List[List[float]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    image_width: int = 1920
    image_height: int = 1080
    focal_length_x: float = 1000.0
    focal_length_y: float = 1000.0
    principal_point_x: float = 960.0
    principal_point_y: float = 540.0
    distortion_coefficients: List[float] = field(default_factory=list)
    projection_type: str = "perspective"

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.CAMERA
        super().__post_init__()

    @property
    def K(self) -> np.ndarray:
        return np.array(self.intrinsic_matrix, dtype=np.float32)

    @property
    def RT(self) -> np.ndarray:
        return np.array(self.extrinsic_matrix, dtype=np.float32)

    @classmethod
    def from_matrices(
        cls,
        K: np.ndarray,
        RT: np.ndarray,
        image_size: Tuple[int, int] = (1920, 1080),
        name: str = "camera",
    ) -> Camera:
        """Construct a Camera contract from NumPy arrays."""
        if K.shape != (3, 3) or RT.shape != (4, 4):
            raise SchemaValidationError(f"Invalid camera matrix shapes: K={K.shape}, RT={RT.shape}")
        return cls(
            name=name,
            intrinsic_matrix=K.tolist(),
            extrinsic_matrix=RT.tolist(),
            image_width=image_size[0],
            image_height=image_size[1],
            focal_length_x=float(K[0, 0]),
            focal_length_y=float(K[1, 1]),
            principal_point_x=float(K[0, 2]),
            principal_point_y=float(K[1, 2]),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "intrinsic_matrix": self.intrinsic_matrix,
            "extrinsic_matrix": self.extrinsic_matrix,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "focal_length_x": self.focal_length_x,
            "focal_length_y": self.focal_length_y,
            "principal_point_x": self.principal_point_x,
            "principal_point_y": self.principal_point_y,
            "distortion_coefficients": self.distortion_coefficients,
            "projection_type": self.projection_type,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Camera:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.CAMERA
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        return cls(**kwargs)


@dataclass
class CameraTrajectory(BaseArtifact):
    """Time-indexed sequence of calibrated camera poses."""

    cameras: List[Camera] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    is_closed_loop: bool = False
    trajectory_smoothness_score: float = 1.0

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.CAMERA_TRAJECTORY
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "cameras": [c.to_dict() for c in self.cameras],
            "timestamps": self.timestamps,
            "is_closed_loop": self.is_closed_loop,
            "trajectory_smoothness_score": self.trajectory_smoothness_score,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CameraTrajectory:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.CAMERA_TRAJECTORY
        if "cameras" in kwargs:
            kwargs["cameras"] = [Camera.from_dict(c) for c in kwargs["cameras"]]
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        return cls(**kwargs)


@dataclass
class Observation(BaseArtifact):
    """Paired raw image observation with associated camera geometry."""

    frame_id: str = ""
    camera: Optional[Camera] = None
    image_uri: Optional[str] = None
    resolution: Tuple[int, int] = (0, 0)
    is_primary_view: bool = False

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.OBSERVATION
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "frame_id": self.frame_id,
            "camera": self.camera.to_dict() if self.camera else None,
            "image_uri": self.image_uri,
            "resolution": list(self.resolution),
            "is_primary_view": self.is_primary_view,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Observation:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.OBSERVATION
        if "camera" in kwargs and kwargs["camera"]:
            kwargs["camera"] = Camera.from_dict(kwargs["camera"])
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        if "resolution" in kwargs and isinstance(kwargs["resolution"], list):
            kwargs["resolution"] = tuple(kwargs["resolution"])
        return cls(**kwargs)


@dataclass
class DepthMap(BaseArtifact):
    """Dense depth map output from sensor or perception models."""

    width: int = 0
    height: int = 0
    min_depth: float = 0.0
    max_depth: float = 1.0
    is_metric: bool = False
    storage_path: Optional[str] = None
    camera_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.DEPTH_MAP
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "width": self.width,
            "height": self.height,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "is_metric": self.is_metric,
            "storage_path": self.storage_path,
            "camera_id": self.camera_id,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DepthMap:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.DEPTH_MAP
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        return cls(**kwargs)


@dataclass
class SegmentationMask(BaseArtifact):
    """Panoptic, semantic, or instance masks with class metadata."""

    width: int = 0
    height: int = 0
    num_instances: int = 0
    class_labels: List[str] = field(default_factory=list)
    storage_path: Optional[str] = None
    frame_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.SEGMENTATION_MASK
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "width": self.width,
            "height": self.height,
            "num_instances": self.num_instances,
            "class_labels": self.class_labels,
            "storage_path": self.storage_path,
            "frame_id": self.frame_id,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SegmentationMask:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.SEGMENTATION_MASK
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        return cls(**kwargs)


@dataclass
class ReconstructionHypothesis(BaseArtifact):
    """Candidate 3D shape hypothesis produced by generative 3D workers."""

    worker_name: str = ""  # TripoSR, TRELLIS, Hunyuan3D
    mesh_artifact_id: Optional[str] = None
    geometric_evidence_agreement_score: float = 0.0
    completeness_score: float = 0.0
    surface_smoothness_score: float = 0.0
    inference_seconds: float = 0.0
    ranking_score: float = 0.0

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.RECONSTRUCTION_HYPOTHESIS
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "worker_name": self.worker_name,
            "mesh_artifact_id": self.mesh_artifact_id,
            "geometric_evidence_agreement_score": self.geometric_evidence_agreement_score,
            "completeness_score": self.completeness_score,
            "surface_smoothness_score": self.surface_smoothness_score,
            "inference_seconds": self.inference_seconds,
            "ranking_score": self.ranking_score,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReconstructionHypothesis:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.RECONSTRUCTION_HYPOTHESIS
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        return cls(**kwargs)


@dataclass
class ConfidenceMap(BaseArtifact):
    """Multi-factor spatial confidence matrix."""

    dimensions: Tuple[int, ...] = (0, 0)
    geometry_confidence: float = 1.0
    camera_confidence: float = 1.0
    texture_confidence: float = 1.0
    semantic_confidence: float = 1.0
    temporal_confidence: float = 1.0
    storage_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.CONFIDENCE_MAP
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "dimensions": list(self.dimensions),
            "geometry_confidence": self.geometry_confidence,
            "camera_confidence": self.camera_confidence,
            "texture_confidence": self.texture_confidence,
            "semantic_confidence": self.semantic_confidence,
            "temporal_confidence": self.temporal_confidence,
            "storage_path": self.storage_path,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConfidenceMap:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.CONFIDENCE_MAP
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        if "dimensions" in kwargs and isinstance(kwargs["dimensions"], list):
            kwargs["dimensions"] = tuple(kwargs["dimensions"])
        return cls(**kwargs)


@dataclass
class ValidationReport(BaseArtifact):
    """Comprehensive validation assessment."""

    is_valid: bool = True
    overall_quality_score: float = 1.0
    camera_pose_stability: float = 1.0
    reprojection_error_pixels: float = 0.0
    is_watertight: bool = True
    is_manifold: bool = True
    num_holes_detected: int = 0
    num_degenerate_faces: int = 0
    uv_overlap_ratio: float = 0.0
    texture_seam_discontinuity: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommended_replanning_actions: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.VALIDATION_REPORT
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "is_valid": self.is_valid,
            "overall_quality_score": self.overall_quality_score,
            "camera_pose_stability": self.camera_pose_stability,
            "reprojection_error_pixels": self.reprojection_error_pixels,
            "is_watertight": self.is_watertight,
            "is_manifold": self.is_manifold,
            "num_holes_detected": self.num_holes_detected,
            "num_degenerate_faces": self.num_degenerate_faces,
            "uv_overlap_ratio": self.uv_overlap_ratio,
            "texture_seam_discontinuity": self.texture_seam_discontinuity,
            "warnings": self.warnings,
            "errors": self.errors,
            "recommended_replanning_actions": self.recommended_replanning_actions,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationReport:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.VALIDATION_REPORT
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        return cls(**kwargs)
