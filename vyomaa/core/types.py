"""Enumerations and strong typing definitions for the VYOMAA Engine."""

from __future__ import annotations

from enum import Enum, auto


class ArtifactType(str, Enum):
    """Canonical artifact types across all engine subsystems."""
    INPUT_MEDIA = "input_media"
    MEDIA_SEQUENCE = "media_sequence"
    FRAME = "frame"
    OBSERVATION = "observation"
    CAMERA = "camera"
    CAMERA_TRAJECTORY = "camera_trajectory"
    DEPTH_MAP = "depth_map"
    SEGMENTATION_MASK = "segmentation_mask"
    POINT_CLOUD = "point_cloud"
    MESH = "mesh"
    GAUSSIAN_SPLAT = "gaussian_splat"
    SDF_VOLUME = "sdf_volume"
    PBR_MATERIAL = "pbr_material"
    TEXTURE = "texture"
    OBJECT_ENTITY = "object_entity"
    SCENE_GRAPH = "scene_graph"
    WORLD_GRAPH = "world_graph"
    RECONSTRUCTION_HYPOTHESIS = "reconstruction_hypothesis"
    CONFIDENCE_MAP = "confidence_map"
    VALIDATION_REPORT = "validation_report"


class ModalityType(str, Enum):
    """Input sensor modality classes."""
    RGB_IMAGE = "rgb_image"
    MULTIVIEW_IMAGE_SET = "multiview_image_set"
    MONOCULAR_VIDEO = "monocular_video"
    MULTI_CAMERA_VIDEO = "multi_camera_video"
    RGBD_IMAGE = "rgbd_image"
    RGBD_VIDEO = "rgbd_video"


class ConfidenceLevel(str, Enum):
    """Standard qualitative confidence ranking."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    GROUND_TRUTH = "ground_truth"


class TaskState(str, Enum):
    """Status lifecycle of a pipeline task."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DeviceType(str, Enum):
    """Target hardware compute device."""
    CPU = "cpu"
    CUDA = "cuda"


class PrecisionType(str, Enum):
    """Floating point precision mode."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class ModelRuntimeState(str, Enum):
    """VRAM residency state of a model adapter."""
    UNINITIALIZED = "uninitialized"
    LOADED_RESIDENT = "loaded_resident"
    ACTIVE = "active"
    SLEEPING = "sleeping"
    EVICTED = "evicted"
    ERROR = "error"


class ModelCapability(str, Enum):
    """Functional capabilities of specialized model workers."""
    FOREGROUND_SEGMENTATION = "foreground_segmentation"
    PANOPTIC_SEGMENTATION = "panoptic_segmentation"
    MONOCULAR_DEPTH = "monocular_depth"
    OBJECT_TRACKING = "object_tracking"
    SFM_CALIBRATION = "sfm_calibration"
    DENSE_CORRESPONDENCE = "dense_correspondence"
    NOVEL_VIEW_SYNTHESIS = "novel_view_synthesis"
    RAPID_HYPOTHESIS_GENERATION = "rapid_hypothesis_generation"
    HIGH_DETAIL_3D_GENERATION = "high_detail_3d_generation"
    PBR_ESTIMATION = "pbr_estimation"
    TEXTURE_BAKING = "texture_baking"
    TOPOLOGY_REPAIR = "topology_repair"
    SUPER_RESOLUTION = "super_resolution"
    WORLD_LAYOUT_GENERATION = "world_layout_generation"


class EntityType(str, Enum):
    """Semantic entity categories within a scene or world."""
    STATIC_OBJECT = "static_object"
    DYNAMIC_OBJECT = "dynamic_object"
    TERRAIN = "terrain"
    ARCHITECTURE = "architecture"
    CAMERA_RIG = "camera_rig"
    LIGHT_SOURCE = "light_source"
    AGENT = "agent"
    UNKNOWN = "unknown"


class SpatialRelation(str, Enum):
    """Semantic spatial relationships between entities."""
    CONTAINS = "contains"
    CONTAINED_IN = "contained_in"
    ADJACENT_TO = "adjacent_to"
    SUPPORTS = "supports"
    SUPPORTED_BY = "supported_by"
    ABOVE = "above"
    BELOW = "below"
    ATTACHED_TO = "attached_to"
