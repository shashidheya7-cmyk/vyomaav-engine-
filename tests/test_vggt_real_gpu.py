import unittest
import tempfile
import os
from pathlib import Path
import numpy as np
import torch
import cv2

from vyomaa.multiview.contracts import ViewSet, GeometryEvidence
from vyomaa.camera_geometry.vggt_adapter import VGGTAdapter, ModelUnavailableError
from vyomaa.camera_geometry.analytic_fallback import AnalyticFallbackAdapter
from vyomaa.camera_geometry.geometry_router import GeometryRouter

class TestVGGTRealGPU(unittest.TestCase):
    def test_vggt_unavailable_state_and_dummy_rejection(self):
        config_missing = {"checkpoint_path": "checkpoints/non_existent_weights.pt", "use_cuda": False}
        adapter_missing = VGGTAdapter(config_missing)
        self.assertFalse(adapter_missing.verify_checkpoint())
        with self.assertRaises(FileNotFoundError):
            adapter_missing.initialize()

    def test_vggt_real_gpu_inference(self):
        ckpt_path = Path("checkpoints/vggt_pretrained.pt")
        if not ckpt_path.exists() or not torch.cuda.is_available():
            print("\n[SKIP] Real VGGT GPU smoke test skipped: Pretrained checkpoint not on disk or CUDA unavailable.")
            return

        print("\n==================================================")
        print(" 📐 RUNNING REAL PRETRAINED VGGT CUDA INFERENCE SMOKE TEST")
        print("==================================================")

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            image_paths = []
            for i in range(5):
                img = np.ones((512, 512, 3), dtype=np.uint8) * (180 + i * 15)
                cv2.circle(img, (200 + i * 20, 256), 40, (0, 0, 255), -1)
                p = frames_dir / f"frame_{i}.jpg"
                cv2.imwrite(str(p), img)
                image_paths.append(str(p))

            view_set = ViewSet(
                observation_ids=[f"obs_{i}" for i in range(5)],
                timestamps=[float(i) * 0.1 for i in range(5)],
                keyframe_flags=[True] * 5,
                image_paths=image_paths
            )

            config = {
                "checkpoint_path": str(ckpt_path),
                "use_cuda": True,
                "device_id": 0
            }

            adapter = VGGTAdapter(config)
            self.assertTrue(adapter.verify_checkpoint())
            adapter.initialize()

            self.assertGreater(adapter.parameter_count, 0)
            evidence = adapter.estimate_geometry(view_set)
            self.assertEqual(evidence.backend, "VGGT")
            self.assertEqual(len(evidence.cameras), 5)
            self.assertEqual(evidence.provenance["execution_mode"], "real_model_forward")
            adapter.release()

    def test_router_with_analytic_fallback_label(self):
        fallback = AnalyticFallbackAdapter({})
        router = GeometryRouter({"analytic_fallback": fallback})
        vs = ViewSet(observation_ids=["v1", "v2"], timestamps=[0.0, 1.0], keyframe_flags=[True, True], image_paths=[])
        evidence = router.route(vs)
        self.assertEqual(evidence.backend, "analytic_fallback")
        self.assertNotEqual(evidence.backend, "VGGT")

if __name__ == "__main__":
    unittest.main()
