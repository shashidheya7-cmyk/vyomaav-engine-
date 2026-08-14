echo "=================================================="
echo " 👁️ IMPLEMENTING REAL SAM2 PERCEPTION WORKER"
echo "==================================================\n"

mkdir -p vyomaa/representations
mkdir -p vyomaa/perception
mkdir -p tests
mkdir -p checkpoints

# 1. Write SegmentationSet representation
cat << 'EOT' > vyomaa/representations/segmentation.py
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

@dataclass
class SegmentationMask:
    mask_id: str
    object_id: str
    mask_array: np.ndarray  # bool or uint8 HxW
    confidence: float
    bbox: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask_id": self.mask_id,
            "object_id": self.object_id,
            "mask_shape": list(self.mask_array.shape),
            "confidence": self.confidence,
            "bbox": self.bbox,
            "metadata": self.metadata
        }

@dataclass
class SegmentationSet:
    observation_id: str
    masks: List[SegmentationMask] = field(default_factory=list)
    tracked_ids: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "masks": [m.to_dict() for m in self.masks],
            "tracked_ids": self.tracked_ids,
            "provenance": self.provenance
        }
EOT
echo "[✓] Created vyomaa/representations/segmentation.py"

# 2. Write ArtifactStore helper if not present
cat << 'EOT' > vyomaa/core/artifact_store.py
import json
from pathlib import Path
from typing import Dict, Any

class ArtifactStore:
    def __init__(self, base_dir: str = "export_package"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, artifact_name: str, data: Dict[str, Any]) -> str:
        path = self.base_dir / f"{artifact_name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return str(path)

    def load(self, artifact_name: str) -> Dict[str, Any]:
        path = self.base_dir / f"{artifact_name}.json"
        with open(path, "r") as f:
            return json.load(f)
EOT
echo "[✓] Created vyomaa/core/artifact_store.py"

# 3. Write real SAM2 worker implementation
cat << 'EOT' > vyomaa/perception/sam2_worker.py
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import numpy as np

from vyomaa.representations.segmentation import SegmentationSet, SegmentationMask
from vyomaa.core.artifact_store import ArtifactStore

logger = logging.getLogger("vyomaa.perception.sam2_worker")

class SAM2Worker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checkpoint_path = config.get("checkpoint_path", "checkpoints/sam2.1_hiera_large.pt")
        self.model_cfg = config.get("model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")
        self.device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = config.get("use_cuda", True)

        if self.use_cuda and self.device_str == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA explicitly requested for SAM2 worker, but torch.cuda.is_available() is False.")
            self.device = torch.device("cuda")
        elif not self.use_cuda:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.device_str)

        self.model = None
        self.predictor = None
        self.video_predictor = None
        self._initialized = False

    def verify_checkpoint(self) -> bool:
        path = Path(self.checkpoint_path)
        if not path.exists():
            logger.warning(f"SAM2 checkpoint not found at {path.absolute()}")
            return False
        return True

    def initialize(self) -> bool:
        if self._initialized:
            return True

        if not self.verify_checkpoint():
            raise FileNotFoundError(f"SAM2 checkpoint mandatory for real inference not found at: {self.checkpoint_path}")

        try:
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            logger.info(f"Loading real SAM2 model config {self.model_cfg} with checkpoint {self.checkpoint_path} on {self.device}")
            
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

            self.model = build_sam2(self.model_cfg, self.checkpoint_path, device=self.device, apply_postprocessing=False)
            self.predictor = SAM2ImagePredictor(self.model)
            try:
                self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint_path, device=self.device)
            except Exception as e:
                logger.warning(f"Video predictor build warning: {e}")

            self._initialized = True
            return True
        except ImportError as e:
            raise ImportError(f"Official 'sam2' package is not installed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SAM2 model: {e}")

    def segment_image(
        self,
        image_path: str,
        observation_id: str,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        artifact_store: Optional[ArtifactStore] = None
    ) -> SegmentationSet:
        if not self._initialized:
            self.initialize()

        start_time = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        import cv2
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image from path: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image)

        masks_list = []
        tracked_ids = []

        if point_coords is not None and len(point_coords) > 0:
            if point_labels is None:
                point_labels = np.ones(len(point_coords), dtype=np.int32)
            
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )

            for idx, (mask, score) in enumerate(zip(masks, scores)):
                mask_id = f"{observation_id}_mask_{idx}"
                obj_id = f"obj_{idx}"
                tracked_ids.append(obj_id)
                
                y_indices, x_indices = np.where(mask)
                bbox = []
                if len(x_indices) > 0 and len(y_indices) > 0:
                    bbox = [int(x_indices.min()), int(y_indices.min()), int(x_indices.max()), int(y_indices.max())]

                masks_list.append(SegmentationMask(
                    mask_id=mask_id,
                    object_id=obj_id,
                    mask_array=mask,
                    confidence=float(score),
                    bbox=bbox,
                    metadata={"source_image": image_path}
                ))

        elapsed = time.time() - start_time
        vram_alloc = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
        vram_peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0

        provenance = {
            "worker": "SAM2Worker",
            "checkpoint": self.checkpoint_path,
            "device": str(self.device),
            "inference_latency_sec": elapsed,
            "vram_allocated_bytes": vram_alloc,
            "vram_peak_bytes": vram_peak,
            "timestamp": time.time()
        }

        seg_set = SegmentationSet(
            observation_id=observation_id,
            masks=masks_list,
            tracked_ids=list(set(tracked_ids)),
            provenance=provenance
        )

        if artifact_store is not None:
            artifact_store.save(f"segmentation_{observation_id}", seg_set.to_dict())

        return seg_set

    def capabilities(self) -> Dict[str, Any]:
        return {
            "worker": "SAM2Worker",
            "initialized": self._initialized,
            "checkpoint": self.checkpoint_path,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available()
        }
EOT
echo "[✓] Created vyomaa/perception/sam2_worker.py"

# 4. Write Comprehensive Unit & Integration Tests: tests/test_sam2_worker.py
cat << 'EOT' > tests/test_sam2_worker.py
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
EOT
echo "[✓] Created tests/test_sam2_worker.py"

echo "\n=================================================="
echo " 🧪 RUNNING SAM2 WORKER TEST SUITE..."
echo "==================================================s"
/root/miniconda3/envs/py3.10/bin/python3 -m unittest tests/test_sam2_worker.py
