"""
VYOMAAV Base Model Engine
Module: world_language.ast

Defines the Abstract Syntax Tree (AST) node hierarchy for World Language (WL) v1.0.
Every node retains line, column, and character offset spans to map compiler diagnostics
and semantic analysis back to the exact source location.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ASTNode:
    """Base abstract class for all World Language AST nodes."""
    line: int
    column: int
    start_offset: int
    end_offset: int


# --- Root & Top-Level Declarations ---

@dataclass
class ProgramNode(ASTNode):
    """Root node representing a complete WL source document."""
    world_state: "WorldStateNode"


@dataclass
class WorldStateNode(ASTNode):
    """Represents a 'world_state' block containing cameras, entities, and environment."""
    name: str
    blocks: List[ASTNode] = field(default_factory=list)


# --- Camera Trajectory ---

@dataclass
class CameraFrameNode(ASTNode):
    """Represents an individual camera view frame within a trajectory."""
    frame_id: str
    pose_se3: List[float]
    intrinsics_k: List[float]
    distortion: Optional[List[float]] = None
    fov: Optional[float] = None


@dataclass
class CameraTrajectoryNode(ASTNode):
    """Represents a 'camera_trajectory' container block."""
    frames: List[CameraFrameNode] = field(default_factory=list)


# --- Entity Component Nodes ---

@dataclass
class SemanticComponentNode(ASTNode):
    """Semantic taxonomy and classification attributes."""
    label: str
    class_id: int
    confidence: float


@dataclass
class SpatialComponentNode(ASTNode):
    """Geometric orientation, bounding volumes, and implicit SDF references."""
    bbox_min: List[float]
    bbox_max: List[float]
    transform_matrix: Optional[List[float]] = None
    sdf_latent_ref: Optional[str] = None


@dataclass
class MaterialComponentNode(ASTNode):
    """PBR visual material parameters."""
    base_type: str
    roughness: float
    metallic: float
    albedo_rgb: Optional[List[float]] = None
    normal_map_ref: Optional[str] = None


@dataclass
class PhysicsComponentNode(ASTNode):
    """Physical dynamics and collision response attributes."""
    mass_kg: float
    friction: float
    is_static: bool
    restitution: Optional[float] = None


@dataclass
class AffordanceComponentNode(ASTNode):
    """Action capabilities and physical interaction boundaries."""
    actions: List[str]
    max_load_kg: Optional[float] = None


@dataclass
class RelationPairNode(ASTNode):
    """Single spatial/topological edge pointing to a target entity."""
    relation_type: str
    target_entity_id: str


@dataclass
class RelationshipComponentNode(ASTNode):
    """Container for entity relationship pairs."""
    relations: List[RelationPairNode] = field(default_factory=list)


@dataclass
class UncertaintyComponentNode(ASTNode):
    """Aleatoric observation noise and epistemic completion risk."""
    aleatoric_noise: float
    epistemic_risk: float
    is_inferred: Optional[bool] = None


@dataclass
class DynamicsComponentNode(ASTNode):
    """Linear and angular velocity vectors."""
    linear_velocity: List[float]
    angular_velocity: List[float]


@dataclass
class EntityNode(ASTNode):
    """Represents a discrete physical or virtual entity within the scene."""
    entity_id: str
    semantic: Optional[SemanticComponentNode] = None
    spatial: Optional[SpatialComponentNode] = None
    material: Optional[MaterialComponentNode] = None
    physics: Optional[PhysicsComponentNode] = None
    affordances: Optional[AffordanceComponentNode] = None
    relationships: Optional[RelationshipComponentNode] = None
    uncertainty: Optional[UncertaintyComponentNode] = None
    dynamics: Optional[DynamicsComponentNode] = None


# --- Environment ---

@dataclass
class EnvironmentNode(ASTNode):
    """Represents background lighting and HDRI skybox settings."""
    hdri_ref: str
    ambient_intensity: float