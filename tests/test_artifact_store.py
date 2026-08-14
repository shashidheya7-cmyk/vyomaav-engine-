"""Unit tests for ArtifactStore, manifest generation, checksums, and lineage tracing."""

import shutil
import tempfile
import unittest

from vyomaa.core.contracts import InputArtifact, DepthMap
from vyomaa.core.types import ModalityType
from vyomaa.storage.artifact_store import ArtifactStore


class TestArtifactStore(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.store = ArtifactStore(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_load_artifact(self):
        art = InputArtifact(
            name="Drone Video Frame",
            modality=ModalityType.RGB_IMAGE,
            file_path="/data/frame_001.png",
            resolution=(3840, 2160),
        )
        manifest = self.store.save_artifact(art)
        self.assertEqual(manifest.artifact_id, art.artifact_id)
        self.assertTrue(len(manifest.checksum_sha256) == 64)

        # Load back
        loaded = self.store.load_artifact(art.artifact_id, InputArtifact)
        self.assertEqual(loaded.artifact_id, art.artifact_id)
        self.assertEqual(loaded.name, "Drone Video Frame")
        self.assertEqual(loaded.resolution, (3840, 2160))

    def test_lineage_tracking(self):
        parent = InputArtifact(name="Parent Input")
        self.store.save_artifact(parent)

        child = DepthMap(name="Child Depth")
        child.provenance.add_parent(parent.artifact_id)
        self.store.save_artifact(child)

        lineage = self.store.get_lineage(child.artifact_id)
        self.assertIn(parent.artifact_id, lineage)


if __name__ == "__main__":
    unittest.main()
