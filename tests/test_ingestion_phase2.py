"""Unit tests for Phase 2 media ingestion subsystem."""

import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np
from PIL import Image

from vyomaa.core.types import ModalityType
from vyomaa.ingestion.image_loader import ImageLoader
from vyomaa.ingestion.metadata_extractor import MetadataExtractor
from vyomaa.ingestion.rgbd_loader import RGBDLoader
from vyomaa.ingestion.video_processor import VideoProcessor


class TestIngestionPhase2(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Create a sample test image
        self.img_path = os.path.join(self.test_dir, "sample.png")
        img = Image.new("RGB", (640, 480), color=(120, 180, 240))
        img.save(self.img_path)

        # Create a multi-view sample directory
        self.mv_dir = os.path.join(self.test_dir, "mv_images")
        os.makedirs(self.mv_dir, exist_ok=True)
        for i in range(3):
            img_i = Image.new("RGB", (320, 240), color=(50 * i, 70 * i, 100))
            img_i.save(os.path.join(self.mv_dir, f"view_{i:02d}.png"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_single_image_loader(self):
        input_art, obs = ImageLoader.load_single_image(self.img_path)
        self.assertEqual(input_art.modality, ModalityType.RGB_IMAGE)
        self.assertEqual(input_art.resolution, (640, 480))
        self.assertIsNotNone(obs.camera)
        self.assertEqual(obs.resolution, (640, 480))

    def test_multiview_directory_loader(self):
        input_art, obs_list = ImageLoader.load_multiview_directory(self.mv_dir)
        self.assertEqual(input_art.modality, ModalityType.MULTIVIEW_IMAGE_SET)
        self.assertEqual(len(obs_list), 3)
        self.assertEqual(obs_list[0].resolution, (320, 240))

    def test_metadata_extractor(self):
        meta = MetadataExtractor.extract_image_exif(self.img_path)
        self.assertEqual(meta["width"], 640)
        self.assertEqual(meta["height"], 480)
        self.assertEqual(meta["format"], "PNG")


if __name__ == "__main__":
    unittest.main()
