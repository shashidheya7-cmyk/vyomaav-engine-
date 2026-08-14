"""SOMG Spatial Object Memory Graph Entity Definitions."""
from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class SemanticComponent:
    label: str = "unknown"
    class_id: int = 0
    confidence: float = 1.0

@dataclass
class SpatialComponent:
    bbox_min: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bbox_max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    transform_matrix: Optional[List[float]] = None
    sdf_ref: Optional[str] = None

@dataclass
class PhysicsComponent:
    mass_kg: float = 1.0
    material: str = "generic"
    is_static: bool = False

@dataclass
class MaterialComponent:
    albedo_color: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8])
    roughness: float = 0.5
    metallic: float = 0.0
    opacity: float = 1.0
    pbr_material_type: str = "generic"
    material_type: str = "generic"

    def __post_init__(self):
        if self.material_type != "generic" and self.pbr_material_type == "generic":
            self.pbr_material_type = self.material_type
        elif self.pbr_material_type != "generic" and self.material_type == "generic":
            self.material_type = self.pbr_material_type

@dataclass
class UncertaintyComponent:
    aleatoric_noise: float = 0.05
    epistemic_risk: float = 0.1
    is_inferred: bool = False

@dataclass
class SOMGEntity:
    entity_id: str
    version: int = 1
    semantic: SemanticComponent = field(default_factory=SemanticComponent)
    spatial: SpatialComponent = field(default_factory=SpatialComponent)
    physics: PhysicsComponent = field(default_factory=PhysicsComponent)
    material: MaterialComponent = field(default_factory=MaterialComponent)
    uncertainty: UncertaintyComponent = field(default_factory=UncertaintyComponent)

    def increment_version(self) -> "SOMGEntity":
        return SOMGEntity(
            entity_id=self.entity_id,
            version=self.version + 1,
            semantic=self.semantic,
            spatial=self.spatial,
            physics=self.physics,
            material=self.material,
            uncertainty=self.uncertainty
        )
