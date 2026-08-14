"""Structured, validated configuration architecture for the VYOMAA Engine."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from .exceptions import ConfigurationError
from .types import DeviceType, PrecisionType


@dataclass
class HardwareConfig:
    device: str = "cuda"
    precision: str = "fp16"
    vram_budget_gb: float = 85.0
    reserve_headroom_gb: float = 6.0
    cuda_device_index: int = 0

    def validate(self) -> None:
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ConfigurationError(f"Unsupported precision '{self.precision}'")
        if self.vram_budget_gb <= 0:
            raise ConfigurationError("vram_budget_gb must be positive")


@dataclass
class IngestionConfig:
    max_resolution: List[int] = field(default_factory=lambda: [3840, 2160])
    max_video_frames: int = 600
    target_fps: float = 30.0
    color_space: str = "sRGB"


@dataclass
class PerceptionConfig:
    depth_model: str = "depth_anything_v2_vitl"
    segmentation_model: str = "sam2_hiera_large"
    tracking_model: str = "bytetrack"
    foreground_threshold: float = 0.5
    batch_size: int = 8


@dataclass
class CameraGeometryConfig:
    geometry_backend: str = "mast3r"
    colmap_enabled: bool = True
    bundle_adjustment_iterations: int = 100
    min_triangulation_angle_deg: float = 2.0


@dataclass
class Worker3DConfig:
    primary_worker: str = "TRELLIS"  # TripoSR, TRELLIS, Hunyuan3D
    triposr_model_id: str = "stabilityai/TripoSR"
    trellis_model_id: str = "microsoft/TRELLIS-image-large"
    hunyuan3d_model_id: str = "tencent/Hunyuan3D-2"
    sparse_steps: int = 12
    slat_steps: int = 12


@dataclass
class FusionConfig:
    evidence_weight: float = 0.85
    generative_prior_weight: float = 0.15
    confidence_floor: float = 0.4
    tsdf_voxel_size: float = 0.005


@dataclass
class TextureMaterialConfig:
    texture_resolution: int = 2048
    bake_with_nvdiffrast: bool = True
    uv_padding_texels: int = 8
    default_roughness: float = 0.5
    default_metallic: float = 0.0


@dataclass
class RefinementConfig:
    target_face_count: int = 40000
    manifold_repair_enabled: bool = True
    qem_decimation_enabled: bool = True
    lod_levels: List[int] = field(default_factory=lambda: [40000, 20000, 10000, 5000])


@dataclass
class ValidationConfig:
    min_overall_confidence: float = 0.6
    max_reprojection_error_pixels: float = 2.5
    require_watertight_manifold: bool = True
    require_uv_chart_packing: bool = True


@dataclass
class StorageConfig:
    workspace_root: str = "workspace"
    auto_generate_manifests: bool = True
    verify_checksums_on_load: bool = True


@dataclass
class EngineConfig:
    """Master engine configuration integrating all typed subsystem groups."""

    profile_name: str = "default_blackwell"
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    camera_geometry: CameraGeometryConfig = field(default_factory=CameraGeometryConfig)
    workers_3d: Worker3DConfig = field(default_factory=Worker3DConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    texture_material: TextureMaterialConfig = field(default_factory=TextureMaterialConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def validate(self) -> None:
        self.hardware.validate()
        if self.validation.min_overall_confidence < 0.0 or self.validation.min_overall_confidence > 1.0:
            raise ConfigurationError("validation.min_overall_confidence must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> EngineConfig:
        """Load, parse, and validate an EngineConfig from a YAML file."""
        p = Path(yaml_path)
        if not p.is_file():
            raise ConfigurationError(f"Configuration file not found: {p}")
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            raise ConfigurationError(f"Failed to parse YAML from {p}: {exc}") from exc

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EngineConfig:
        """Construct validated EngineConfig from dictionary."""
        d = dict(data)
        hw = HardwareConfig(**d.get("hardware", {}))
        ing = IngestionConfig(**d.get("ingestion", {}))
        perc = PerceptionConfig(**d.get("perception", {}))
        cam = CameraGeometryConfig(**d.get("camera_geometry", {}))
        w3d = Worker3DConfig(**d.get("workers_3d", {}))
        fus = FusionConfig(**d.get("fusion", {}))
        tm = TextureMaterialConfig(**d.get("texture_material", {}))
        ref = RefinementConfig(**d.get("refinement", {}))
        val = ValidationConfig(**d.get("validation", {}))
        st = StorageConfig(**d.get("storage", {}))

        cfg = cls(
            profile_name=d.get("profile_name", "custom_profile"),
            hardware=hw,
            ingestion=ing,
            perception=perc,
            camera_geometry=cam,
            workers_3d=w3d,
            fusion=fus,
            texture_material=tm,
            refinement=ref,
            validation=val,
            storage=st,
        )
        cfg.validate()
        return cfg
