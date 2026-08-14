"""
VYOMAAV Base Model Engine
Module: world_language.validator

Multi-layered semantic and physical validator for World Language ASTs (Sprint 1.4).
Validates vector dimensions, entity referential integrity, physical constraints,
and topological consistency.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set
from world_language.ast import ProgramNode, EntityNode, CameraTrajectoryNode, ASTNode
from world_language.tokenizer import Diagnostic, DiagnosticSeverity


@dataclass
class ValidationResult:
    is_valid: bool
    diagnostics: List[Diagnostic] = field(default_factory=list)


class WLValidator:
    """Multi-layered semantic and physical validator for WL ASTs."""

    def __init__(self, program: ProgramNode):
        self.program = program
        self.diagnostics: List[Diagnostic] = []
        self.entities: Dict[str, EntityNode] = {}

    def _report(self, message: str, node: ASTNode, severity: DiagnosticSeverity = DiagnosticSeverity.ERROR):
        self.diagnostics.append(
            Diagnostic(message, node.line, node.column, node.start_offset, node.end_offset, severity)
        )

    def validate(self) -> ValidationResult:
        """Executes full multi-layered validation pass over the AST."""
        if not self.program or not self.program.world_state:
            return ValidationResult(is_valid=False, diagnostics=self.diagnostics)

        # Index Entities for Referential Integrity Pass
        for block in self.program.world_state.blocks:
            if isinstance(block, EntityNode):
                if block.entity_id in self.entities:
                    self._report(f"Duplicate entity ID detected: '{block.entity_id}'", block)
                else:
                    self.entities[block.entity_id] = block

        # Run Validation Layers
        for block in self.program.world_state.blocks:
            if isinstance(block, CameraTrajectoryNode):
                self._validate_camera_trajectory(block)
            elif isinstance(block, EntityNode):
                self._validate_entity(block)

        # World Consistency: Check Circular Dependencies
        self._validate_circular_relationships()

        has_errors = any(d.severity == DiagnosticSeverity.ERROR for d in self.diagnostics)
        return ValidationResult(is_valid=not has_errors, diagnostics=self.diagnostics)

    def _validate_camera_trajectory(self, traj: CameraTrajectoryNode):
        for frame in traj.frames:
            if len(frame.pose_se3) != 12:
                self._report(f"Frame '{frame.frame_id}' pose_se3 must have exactly 12 elements, got {len(frame.pose_se3)}", frame)
            if len(frame.intrinsics_k) != 9:
                self._report(f"Frame '{frame.frame_id}' intrinsics_k must have exactly 9 elements, got {len(frame.intrinsics_k)}", frame)
            if frame.fov is not None and (frame.fov <= 0.0 or frame.fov >= 180.0):
                self._report(f"Frame '{frame.frame_id}' FOV must be in (0, 180) degrees, got {frame.fov}", frame)

    def _validate_entity(self, entity: EntityNode):
        # 1. Semantic Check
        if entity.semantic:
            if not entity.semantic.label:
                self._report(f"Entity '{entity.entity_id}' semantic label cannot be empty", entity.semantic)
            if not (0.0 <= entity.semantic.confidence <= 1.0):
                self._report(f"Entity '{entity.entity_id}' confidence must be in [0.0, 1.0], got {entity.semantic.confidence}", entity.semantic)

        # 2. Spatial & Bounding Box Check
        if entity.spatial:
            min_vec = entity.spatial.bbox_min
            max_vec = entity.spatial.bbox_max
            if len(min_vec) != 3 or len(max_vec) != 3:
                self._report(f"Entity '{entity.entity_id}' bbox_min and bbox_max must be 3D vectors", entity.spatial)
            else:
                if min_vec[0] >= max_vec[0] or min_vec[1] >= max_vec[1] or min_vec[2] >= max_vec[2]:
                    self._report(f"Entity '{entity.entity_id}' bbox_min {min_vec} must be strictly less than bbox_max {max_vec}", entity.spatial)

        # 3. Physics Check
        if entity.physics:
            if entity.physics.mass_kg < 0.0:
                self._report(f"Entity '{entity.entity_id}' mass_kg cannot be negative: {entity.physics.mass_kg}", entity.physics)
            if entity.physics.friction < 0.0:
                self._report(f"Entity '{entity.entity_id}' friction cannot be negative: {entity.physics.friction}", entity.physics)
            if entity.physics.restitution is not None and not (0.0 <= entity.physics.restitution <= 1.0):
                self._report(f"Entity '{entity.entity_id}' restitution must be in [0.0, 1.0]", entity.physics)

        # 4. Relationship Referential Integrity Check
        if entity.relationships:
            for rel in entity.relationships.relations:
                if rel.target_entity_id not in self.entities:
                    self._report(
                        f"Entity '{entity.entity_id}' references unknown target entity '{rel.target_entity_id}' in relationship '{rel.relation_type.value}'",
                        rel
                    )

    def _validate_circular_relationships(self):
        """Detect circular dependencies in 'supported_by' relationships."""
        for entity_id, entity in self.entities.items():
            visited: Set[str] = set()
            curr = entity_id
            while curr in self.entities:
                if curr in visited:
                    self._report(f"Circular dependency detected in 'supported_by' relationships involving entity '{entity_id}'", entity)
                    break
                visited.add(curr)
                # Find supported_by target
                target = None
                if self.entities[curr].relationships:
                    for rel in self.entities[curr].relationships.relations:
                        if rel.relation_type.value == "supported_by":
                            target = rel.target_entity_id
                            break
                curr = target