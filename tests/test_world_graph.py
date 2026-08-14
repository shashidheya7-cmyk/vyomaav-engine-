"""Unit tests for WorldGraph, regions, and spatial relations."""

import unittest

from vyomaa.core.types import SpatialRelation
from vyomaa.scene_graph.scene_graph import SceneGraph
from vyomaa.scene_graph.world_graph import WorldGraph, WorldRegion


class TestWorldGraph(unittest.TestCase):

    def test_world_graph_composition(self):
        wg = WorldGraph(name="Global Campus")
        
        region = WorldRegion(name="Building A - Floor 2", is_indoor=True)
        wg.add_region(region)

        sg = SceneGraph(name="Conference Room 201")
        wg.add_scene_graph(sg)
        region.scene_graph_ids.append(sg.artifact_id)

        wg.add_spatial_relation(
            source_id=sg.artifact_id,
            target_id=region.region_id,
            relation=SpatialRelation.CONTAINED_IN,
            confidence=0.99,
        )

        d = wg.to_dict()
        wg_re = WorldGraph.from_dict(d)
        self.assertEqual(len(wg_re.regions), 1)
        self.assertEqual(len(wg_re.scene_graphs), 1)
        self.assertEqual(len(wg_re.spatial_edges), 1)
        self.assertEqual(wg_re.spatial_edges[0].relation, SpatialRelation.CONTAINED_IN)


if __name__ == "__main__":
    unittest.main()
