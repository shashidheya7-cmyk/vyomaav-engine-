"""
VYOMAAV Base Model Engine
Module: somg.exporter

Exports active SOMG scene graphs into OpenUSD spatial specs and JSON-LD payloads
suitable for downstream reconstruction, WebGPU rendering, and simulation engines.
"""

import json
from typing import Dict, Any, List
from somg.scene import SceneState
from somg.entity import SOMGEntity


class SOMGExporter:
    """Exports SceneState graph payloads into engine formats."""

    @staticmethod
    def to_json_ld(scene: SceneState) -> str:
        """Converts active scene state to structured JSON-LD format."""
        graph = scene.resolve_active_graph()
        entities_payload: List[Dict[str, Any]] = []

        for entity in graph.nodes.values():
            e_dict: Dict[str, Any] = {
                "@id": entity.entity_id,
                "@type": "SpatialEntity",
                "version": entity.version,
                "semantic": {
                    "label": entity.semantic.label,
                    "class_id": entity.semantic.class_id,
                    "confidence": entity.semantic.confidence
                },
                "spatial": {
                    "bbox_min": entity.spatial.bbox_min,
                    "bbox_max": entity.spatial.bbox_max
                },
                "physics": {
                    "mass_kg": entity.physics.mass_kg,
                    "is_static": entity.physics.is_static
                },
                "uncertainty": {
                    "aleatoric": entity.uncertainty.aleatoric_noise,
                    "epistemic": entity.uncertainty.epistemic_risk
                },
                "relationships": []
            }

            if entity.entity_id in graph.outgoing_edges:
                for edge in graph.outgoing_edges[entity.entity_id]:
                    e_dict["relationships"].append({
                        "relation": edge.relation_type.value,
                        "target": edge.target_id
                    })

            entities_payload.append(e_dict)

        payload = {
            "@context": "https://vyomaav.ai/contexts/somg-v1.jsonld",
            "scene_id": scene.scene_id,
            "entity_count": len(entities_payload),
            "entities": entities_payload
        }

        return json.dumps(payload, indent=2)

    @staticmethod
    def to_openusd_ascii(scene: SceneState) -> str:
        """Converts active scene state into OpenUSD ASCII (.usda) stage format."""
        graph = scene.resolve_active_graph()
        lines = [
            '#usda 1.0',
            f'def Xform "{scene.scene_id}"',
            '{'
        ]

        for entity in graph.nodes.values():
            lines.append(f'    def Cube "{entity.entity_id}"')
            lines.append('    {')
            lines.append(f'        custom string vyomaav:label = "{entity.semantic.label}"')
            lines.append(f'        custom float vyomaav:confidence = {entity.semantic.confidence}')
            lines.append(f'        custom float3 vyomaav:bbox_min = {tuple(entity.spatial.bbox_min)}')
            lines.append(f'        custom float3 vyomaav:bbox_max = {tuple(entity.spatial.bbox_max)}')
            lines.append(f'        custom float vyomaav:mass_kg = {entity.physics.mass_kg}')
            lines.append('    }')

        lines.append('}')
        return "\n".join(lines)