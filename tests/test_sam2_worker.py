import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch

from vyomaa.representations.segmentation import SegmentationSet, SegmentationMask
from vyomaa.core.artifact_store import ArtifactStore
from vyomaa.perception.sam2_worker import SAM2Worker

class TestSAM2Worker(unittest.TestCase):
    def test_segmentation_serialization(self):
        mask_arr = np.zeros((10, 10), dtype=bool)
        mask_arr[2:5, 2:5] = True
        mask = SegmentationMask(
            mask_id="m1",
            object_id="obj_0",
            mask_array=mask_arr,
            confidence=0.98,
            bbox=[2, 2, 4, 4]
        )
        seg_set = SegmentationSet(
            observation_id="obs_test",
            masks=[mask],
            tracked_ids=["obj_0"],
            provenance={"test": True}
        )
        d = seg_set.to_dict()
        self.assertEqual(d["observation_id"], "obs_test")
        self.assertEqual(len(d["masks"]), 1)
        self.assertEqual(d["masks"][0]["confidence"], 0.98)

    def test_unavailable_state_error(self):
        # Test with a non-existent checkpoint
        config = {
            "checkpoint_path": "checkpoints/non_existent_sam2.pt",
            "use_cuda": False
        }
        worker = SAM2Worker(config)
        self.assertFalse(worker.verify_checkpoint())
        
        with self.assertRaises(FileNotFoundError):
            worker.initialize()

    def test_strict_cuda_enforcement(self):
        # If CUDA is requested but unavailable, should raise RuntimeError
        if not torch.cuda.is_available():
            config = {
                "checkpoint_path": "checkpoints/sam2.1_hiera_large.pt",
                "use_cuda": True,
                "device": "cuda"
            }
            with self.assertRaises(RuntimeError):
                SAM2Worker(config)

    def test_real_gpu_smoke_test(self):
        # Runs ONLY when official checkpoint exists on disk
        ckpt = Path("checkpoints/sam2.1_hiera_large.pt")
        if not ckpt.exists():
            print("\n[SKIP] Real GPU smoke test skipped: SAM2 checkpoint not present on disk.")
            return

        config = {
            "checkpoint_path": str(ckpt),
            "use_cuda": torch.cuda.is_available()
        }
        worker = SAM2Worker(config)
        self.assertTrue(worker.verify_checkpoint())
        worker.initialize()
        caps = worker.capabilities()
        self.assertTrue(caps["initialized"])
        print(f"\n[OK] Real GPU smoke test passed with checkpoint {ckpt}!")

if __name__ == "__main__":
    unittest.main()
