import unittest
import tempfile
import os
from pathlib import Path
import numpy as np
import torch
import cv2
import json

from vyomaa.core.artifact_store import ArtifactStore
from vyomaa.perception.sam2_worker import SAM2Worker, ModelUnavailableError

class TestSAM2MultiFrame(unittest.TestCase):
    def test_multiview_video_propagation(self):
        ckpt_path = Path("checkpoints/sam2.1_hiera_large.pt")
        
        # Skip conditions
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
            print("\n[SKIP] Multi-frame propagation test skipped: CUDA not available.")
            return
        if not ckpt_path.exists():
            print(f"\n[SKIP] Multi-frame propagation test skipped: Checkpoint not found at {ckpt_path}.")
            return
        
        try:
            import sam2
        except ImportError:
            print("\n[SKIP] Multi-frame propagation test skipped: 'sam2' package not installed.")
            return

        print("\n==================================================")
        print(" 🎞️ RUNNING REAL SAM2 MULTI-FRAME VIDEO PROPAGATION")
        print("==================================================")

        # Create a temp directory for a short sequence of 5 consecutive real frames
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir)
            frame_count = 5
            h, w = 512, 512

            for f_idx in range(frame_count):
                # Moving rectangular object across frames
                img = np.ones((h, w, 3), dtype=np.uint8) * 240
                x_offset = f_idx * 15
                cv2.rectangle(img, (128 + x_offset, 128), (256 + x_offset, 256), (0, 255, 0), -1)
                
                frame_path = frames_dir / f"{f_idx:05d}.jpg"
                cv2.imwrite(str(frame_path), img)

            config = {
                "checkpoint_path": str(ckpt_path),
                "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
                "use_cuda": True,
                "device_id": 0
            }

            worker = SAM2Worker(config)
            worker.initialize()

            artifact_store = ArtifactStore("export_package")
            
            # Seed on object center in frame 0
            seed_points = np.array([[192, 192]], dtype=np.float32)
            seed_labels = np.array([1], dtype=np.int32)

            result = worker.propagate_video_sequence(
                video_frames_dir=str(frames_dir),
                observation_id="obs_multiframe_smoke",
                seed_frame_idx=0,
                obj_id=1,
                points=seed_points,
                labels=seed_labels,
                artifact_store=artifact_store
            )

            segments = result["video_segments"]
            obj_ids = result["object_ids"]
            prov = result["provenance"]

            print(f"[✓] Multi-frame propagation completed successfully!")
            print(f"    - Frames propagated: {result['frame_count']}")
            print(f"    - Object IDs tracked: {obj_ids}")
            print(f"    - Propagation Latency: {prov['propagation_latency_ms']:.2f} ms")
            print(f"    - Peak VRAM: {prov['vram_peak_mb']:.2f} MB")

            # Validation requirements
            self.assertEqual(result["frame_count"], frame_count)
            self.assertIn(1, obj_ids)

            for f_idx, frame_masks in segments.items():
                self.assertIn(1, frame_masks)
                mask = frame_masks[1]
                # Validate mask shape matches frame dimensions
                self.assertEqual(mask.shape, (h, w))
                # Validate mask is finite
                self.assertTrue(np.isfinite(mask).all())
                # Validate mask area is valid (> 0)
                area = int(mask.sum())
                self.assertGreater(area, 0)
                print(f"    - Frame {f_idx}: Object 1 mask area = {area} pixels")

            # Generate reports/phase4b/sam2_multiframe.json
            report_dir = Path("reports/phase4b")
            report_dir.mkdir(parents=True, exist_ok=True)

            multiframe_benchmark = {
                "model": "sam2.1_hiera_large",
                "checkpoint": str(ckpt_path),
                "config": config["model_cfg"],
                "device": "cuda:0",
                "input_video_frames": frame_count,
                "frame_resolution": [w, h],
                "seeded_object_id": 1,
                "propagated_object_ids": obj_ids,
                "propagation_frames": result["frame_count"],
                "mean_latency_ms": prov["propagation_latency_ms"] / frame_count,
                "total_latency_ms": prov["propagation_latency_ms"],
                "peak_vram_mb": prov["vram_peak_mb"],
                "finite_output": True,
                "temporal_id_persistence": True,
                "status": "REAL_MULTI_FRAME_VERIFIED"
            }

            with open(report_dir / "sam2_multiframe.json", "w") as f:
                json.dump(multiframe_benchmark, f, indent=2)

            print(f"[✓] Multiframe benchmark report saved to {report_dir / 'sam2_multiframe.json'}")

            worker.release()

    def test_unavailable_state_and_empty(self):
        config = {
            "checkpoint_path": "checkpoints/non_existent.pt",
            "use_cuda": False
        }
        worker = SAM2Worker(config)
        with self.assertRaises(FileNotFoundError):
            worker.initialize()

if __name__ == "__main__":
    unittest.main()
