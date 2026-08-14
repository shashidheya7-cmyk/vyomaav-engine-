"""
VYOMAAV Base Model Engine
Module: somg.optimizer

Compresses long delta stacks into flattened checkpoints (Δ-squashing).
"""

from typing import List
from somg.scene import SceneState, DeltaLayer
from somg.graph import SpatialGraph


class DeltaOptimizer:
    """Optimizes memory overhead by squashing delta stacks into checkpoints."""

    @staticmethod
    def squash_deltas(delta_layers: List[DeltaLayer], new_layer_id: str) -> DeltaLayer:
        """Flattens multiple delta layers into a single consolidated DeltaLayer."""
        squashed = DeltaLayer(layer_id=new_layer_id)

        for delta in delta_layers:
            # Process removals
            for rem_id in delta.removed_entity_ids:
                squashed.removed_entity_ids.add(rem_id)
                squashed.added_entities.pop(rem_id, None)
                squashed.updated_entities.pop(rem_id, None)

            # Process additions
            for add_id, entity in delta.added_entities.items():
                squashed.removed_entity_ids.discard(add_id)
                squashed.added_entities[add_id] = entity

            # Process updates
            for up_id, entity in delta.updated_entities.items():
                if up_id in squashed.added_entities:
                    squashed.added_entities[up_id] = entity
                else:
                    squashed.updated_entities[up_id] = entity

        return squashed

    @staticmethod
    def create_checkpoint(scene: SceneState, checkpoint_name: str) -> SceneState:
        """Consolidates current delta stack directly into a new base SceneState."""
        active_graph = scene.resolve_active_graph()
        checkpoint_scene = SceneState(scene_id=f"{scene.scene_id}_{checkpoint_name}")
        checkpoint_scene.camera_graph = scene.camera_graph

        for entity in active_graph.nodes.values():
            checkpoint_scene.base_graph.add_node(entity)

        for src, edges in active_graph.outgoing_edges.items():
            for edge in edges:
                checkpoint_scene.base_graph.add_edge(edge.source_id, edge.target_id, edge.relation_type)

        return checkpoint_scene