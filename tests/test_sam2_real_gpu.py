import unittest
import tempfile
import os
from pathlib import Path
import numpy as np
import torch
import cv2

from vyomaa.representations.segmentation import SegmentationSet, SegmentationMask
from vyomaa.core.artifact_store import ArtifactStore
from vyomaa.perception.sam2_worker import SAM2Worker, ModelUnavailableError

class TestSAM2RealGPU(unittest.TestCase):
    def test_real_gpu_inference_smoke(self):
        ckpt_path = Path("checkpoints/sam2.1_hiera_large.pt")
        
        # Skip conditions
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
            print("\n[SKIP] Real GPU smoke test skipped: CUDA not available.")
            return
        if not ckpt_path.exists():
            print(f"\n[SKIP] Real GPU smoke test skipped: Checkpoint not found at {ckpt_path}.")
            return
        
        try:
            import sam2
        except ImportError:
            print("\n[SKIP] Real GPU smoke test skipped: 'sam2' package not installed.")
            return

        print("\n==================================================")
        print(" 🔬 RUNNING REAL SAM2 CUDA GPU SMOKE TEST")
        print("==================================================")

        # Create a real test image (512x512 RGB with a centered colored square)
        img = np.ones((512, 512, 3), dtype=np.uint8) * 240
        cv2.rectangle(img, (128, 128), (384, 384), (0, 0, 255), -1)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, img)

        try:
            config = {
                "checkpoint_path": str(ckpt_path),
                "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
                "use_cuda": True,
                "device_id": 0
            }

            vram_before = torch.cuda.memory_allocated() / (1024 * 1024)
            worker = SAM2Worker(config)
            
            self.assertTrue(worker.verify_checkpoint())
            worker.initialize()
            caps = worker.capabilities()
            
            print(f"[*] GPU Name: {caps['gpu_name']}")
            print(f"[*] VRAM Before Load: {vram_before:.2f} MB")
            print(f"[*] VRAM After Load: {worker.vram_after_load_mb:.2f} MB")

            # Run real segmentation
            artifact_store = ArtifactStore("export_package")
            seg_set = worker.segment_image(
                image_path=tmp_path,
                observation_id="obs_real_smoke",
                artifact_store=artifact_store
            )

            print(f"[✓] Segmentation completed successfully!")
            print(f"    - Masks generated: {len(seg_set.masks)}")
            print(f"    - Inference latency: {seg_set.provenance['inference_latency_ms']:.2f} ms")
            print(f"    - VRAM Peak: {seg_set.provenance['vram_peak_mb']:.2f} MB")

            # Assertions
            self.assertGreater(len(seg_set.masks), 0)
            for m in seg_set.masks:
                self.assertTrue(np.isfinite(m.mask_array).all())
                self.assertEqual(m.mask_array.shape, (512, 512))

            # Generate benchmark JSON report
            import json
            report_dir = Path("reports/phase4b")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            benchmark_data = {
                "model": "sam2.1_hiera_large",
                "checkpoint": str(ckpt_path),
                "config": config["model_cfg"],
                "device": "cuda:0",
                "gpu_name": caps["gpu_name"],
                "torch_version": caps["torch_version"],
                "torch_cuda": caps["torch_cuda"],
                "input_width": 512,
                "input_height": 512,
                "mask_count": len(seg_set.masks),
                "mask_resolution": [512, 512],
                "inference_ms": seg_set.provenance["inference_latency_ms"],
                "vram_before_mb": vram_before,
                "vram_after_load_mb": worker.vram_after_load_mb,
                "vram_peak_mb": seg_set.provenance["vram_peak_mb"],
                "finite_output": True,
                "checkpoint_exists": True,
                "real_inference": True,
                "status": "REAL_MODEL_INFERENCE_VERIFIED"
            }

            with open(report_dir / "sam2_inference.json", "w") as f:
                json.dump(benchmark_data, f, indent=2)

            print(f"[✓] Benchmark report saved to {report_dir / 'sam2_inference.json'}")

            worker.release()

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()
