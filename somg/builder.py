"""SOMG Perception Observation & Entity Builder."""
from dataclasses import dataclass
from typing import List, Optional
from somg.entity import SOMGEntity, SemanticComponent, SpatialComponent, PhysicsComponent, UncertaintyComponent

@dataclass
class PerceptionObservation:
    observation_id: str
    label: str
    class_id: int
    confidence: float
    bbox_min: List[float]
    bbox_max: List[float]
    estimated_mass_kg: float = 1.0
    material_type: str = "generic"
    aleatoric_noise: float = 0.05
    depth_m: Optional[float] = None
    estimated_depth_m: Optional[float] = None
    obs_id: Optional[str] = None

    def __post_init__(self):
        if self.obs_id is not None:
            self.observation_id = self.obs_id
        if self.depth_m is not None and self.estimated_depth_m is None:
            self.estimated_depth_m = self.depth_m

class SOMGEntityBuilder:
    """Builds SOMG Entities from perception observations."""

    @staticmethod
    def from_observation(obs: PerceptionObservation) -> SOMGEntity:
        return SOMGEntity(
            entity_id=f"entity_{obs.observation_id}",
            semantic=SemanticComponent(
                label=obs.label,
                class_id=obs.class_id,
                confidence=obs.confidence
            ),
            spatial=SpatialComponent(
                bbox_min=obs.bbox_min,
                bbox_max=obs.bbox_max
            ),
            physics=PhysicsComponent(
                mass_kg=obs.estimated_mass_kg,
                material=obs.material_type
            ),
            uncertainty=UncertaintyComponent(
                aleatoric_noise=obs.aleatoric_noise,
                epistemic_risk=0.1,
                is_inferred=False
            )
        )
