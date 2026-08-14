"""Unit tests for SceneGraph hierarchy, search, and global transforms."""

import unittest
import numpy as np

from vyomaa.core.contracts import Camera
from vyomaa.core.exceptions import SceneGraphError
from vyomaa.scene_graph.entity import ObjectEntity
from vyomaa.scene_graph.scene_graph import SceneGraph
from vyomaa.scene_graph.transform import Transform3D


class TestSceneGraph(unittest.TestCase):

    def test_scenegraph_hierarchy_and_global_transform(self):
        sg = SceneGraph(name="Living Room Scene")
        
        # Add a parent entity (Table)
        table = ObjectEntity(
            name="Wooden Table",
            local_transform=Transform3D(translation=[1.0, 0.0, 0.0]),
            semantic_labels=["furniture", "table"],
        )
        sg.add_entity(table)

        # Add a child entity (Cup on Table)
        cup = ObjectEntity(
            name="Coffee Cup",
            local_transform=Transform3D(translation=[0.0, 0.8, 0.0]),
            semantic_labels=["prop", "cup"],
        )
        sg.add_entity(cup, parent_id=table.artifact_id)

        # Verify parent-child links
        self.assertEqual(cup.parent_id, table.artifact_id)
        self.assertIn(cup.artifact_id, table.child_ids)

        # Compute global transform for cup: (1.0, 0.8, 0.0)
        global_t = sg.get_global_transform(cup.artifact_id)
        self.assertAlmostEqual(global_t.translation[0], 1.0)
        self.assertAlmostEqual(global_t.translation[1], 0.8)
        self.assertAlmostEqual(global_t.translation[2], 0.0)

        # Search by label
        cups = sg.find_entities_by_label("cup")
        self.assertEqual(len(cups), 1)
        self.assertEqual(cups[0].artifact_id, cup.artifact_id)

    def test_scenegraph_entity_removal(self):
        sg = SceneGraph()
        e1 = ObjectEntity(name="E1")
        e2 = ObjectEntity(name="E2")
        sg.add_entity(e1)
        sg.add_entity(e2, parent_id=e1.artifact_id)

        # Remove e1 -> e2 re-parented to root
        sg.remove_entity(e1.artifact_id)
        self.assertNotIn(e1.artifact_id, sg.entities)
        self.assertIn(e2.artifact_id, sg.entities)
        self.assertEqual(e2.parent_id, sg.root_id)


if __name__ == "__main__":
    unittest.main()
