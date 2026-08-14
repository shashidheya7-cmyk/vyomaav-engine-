"""Unit tests for configuration profiles and validation."""

import os
import unittest

from vyomaa.core.config import EngineConfig
from vyomaa.core.exceptions import ConfigurationError


class TestConfig(unittest.TestCase):

    def test_load_all_shipped_profiles(self):
        config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
        profiles = [
            "blackwell_rtx6000.yaml",
            "single_image_object.yaml",
            "multiview_object.yaml",
            "video_scene.yaml",
            "world_reconstruction.yaml",
        ]
        for prof in profiles:
            p = os.path.join(config_dir, prof)
            self.assertTrue(os.path.exists(p), f"Profile missing: {p}")
            cfg = EngineConfig.from_yaml(p)
            self.assertIsNotNone(cfg.hardware.device)
            self.assertGreater(cfg.hardware.vram_budget_gb, 0)

    def test_invalid_precision_raises(self):
        with self.assertRaises(ConfigurationError):
            cfg = EngineConfig.from_dict({"hardware": {"precision": "fp8_invalid"}})


if __name__ == "__main__":
    unittest.main()
