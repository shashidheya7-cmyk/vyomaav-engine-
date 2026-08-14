"""
VYOMAAV Base Model Engine
Module: world_language.serializer

Serializes AST nodes back into formatted World Language (WL) text code or JSON (Sprint 1.5).
"""

import json
from typing import Dict, Any
from world_language.ast import ProgramNode, EntityNode, CameraTrajectoryNode, EnvironmentNode


class WLSerializer:
    """Serializes World Language AST nodes into source code or JSON."""

    def __init__(self, indent_spaces: int = 4):
        self.indent_spaces = indent_spaces

    def _indent(self, level: int) -> str:
        return " " * (level * self.indent_spaces)

    def to_wl(self, program: ProgramNode) -> str:
        """Serializes AST to formatted World Language text."""
        if not program or not program.world_state:
            return ""

        ws = program.world_state
        lines = [f'world_state "{ws.name}" {{']

        for block in ws.blocks:
            if isinstance(block, CameraTrajectoryNode):
                lines.append(f'{self._indent(1)}camera_trajectory {{')
                for frame in block.frames:
                    lines.append(f'{self._indent(2)}frame "{frame.frame_id}" {{')
                    lines.append(f'{self._indent(3)}pose_se3: {frame.pose_se3} ;')
                    lines.append(f'{self._indent(3)}intrinsics_k: {frame.intrinsics_k} ;')
                    if frame.fov is not None:
                        lines.append(f'{self._indent(3)}fov: {frame.fov} ;')
                    lines.append(f'{self._indent(2)}}}')
                lines.append(f'{self._indent(1)}}}')

            elif isinstance(block, EntityNode):
                lines.append(f'{self._indent(1)}entity "{block.entity_id}" {{')
                if block.semantic:
                    lines.append(f'{self._indent(2)}semantic {{')
                    lines.append(f'{self._indent(3)}label: "{block.semantic.label}" ;')
                    lines.append(f'{self._indent(3)}class_id: {block.semantic.class_id} ;')
                    lines.append(f'{self._indent(3)}confidence: {block.semantic.confidence} ;')
                    lines.append(f'{self._indent(2)}}}')
                if block.spatial:
                    lines.append(f'{self._indent(2)}spatial {{')
                    lines.append(f'{self._indent(3)}bbox_min: {block.spatial.bbox_min} ;')
                    lines.append(f'{self._indent(3)}bbox_max: {block.spatial.bbox_max} ;')
                    lines.append(f'{self._indent(2)}}}')
                if block.physics:
                    lines.append(f'{self._indent(2)}physics {{')
                    lines.append(f'{self._indent(3)}mass_kg: {block.physics.mass_kg} ;')
                    lines.append(f'{self._indent(3)}friction: {block.physics.friction} ;')
                    lines.append(f'{self._indent(3)}is_static: {"true" if block.physics.is_static else "false"} ;')
                    lines.append(f'{self._indent(2)}}}')
                if block.relationships:
                    lines.append(f'{self._indent(2)}relationships {{')
                    for rel in block.relationships.relations:
                        lines.append(f'{self._indent(3)}{rel.relation_type.value}: "{rel.target_entity_id}" ;')
                    lines.append(f'{self._indent(2)}}}')
                lines.append(f'{self._indent(1)}}}')

            elif isinstance(block, EnvironmentNode):
                lines.append(f'{self._indent(1)}environment {{')
                lines.append(f'{self._indent(2)}hdri_ref: "{block.hdri_ref}" ;')
                lines.append(f'{self._indent(2)}ambient_intensity: {block.ambient_intensity} ;')
                lines.append(f'{self._indent(1)}}}')

        lines.append("}")
        return "\n".join(lines)

    def to_json(self, program: ProgramNode) -> str:
        """Serializes AST to JSON string."""
        if not program or not program.world_state:
            return "{}"

        ws = program.world_state
        payload: Dict[str, Any] = {
            "world_state": ws.name,
            "blocks": []
        }

        for block in ws.blocks:
            if isinstance(block, EntityNode):
                entity_dict: Dict[str, Any] = {"type": "entity", "id": block.entity_id}
                if block.semantic:
                    entity_dict["semantic"] = {"label": block.semantic.label, "confidence": block.semantic.confidence}
                if block.spatial:
                    entity_dict["spatial"] = {"bbox_min": block.spatial.bbox_min, "bbox_max": block.spatial.bbox_max}
                payload["blocks"].append(entity_dict)

        return json.dumps(payload, indent=self.indent_spaces)