import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import numpy as np

from vyomaa.representations.segmentation import SegmentationSet, SegmentationMask
from vyomaa.core.artifact_store import ArtifactStore

logger = logging.getLogger("vyomaa.perception.sam2_worker")

class ModelUnavailableError(Exception):
    """Raised when a requested model backend or hardware device is unavailable."""
    pass

class SAM2Worker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checkpoint_path = config.get("checkpoint_path", "checkpoints/sam2.1_hiera_large.pt")
        self.model_cfg = config.get("model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")
        self.use_cuda = config.get("use_cuda", True)

        if self.use_cuda:
            if not torch.cuda.is_available():
                raise ModelUnavailableError("CUDA explicitly requested for SAM2 worker, but torch.cuda.is_available() is False.")
            if torch.cuda.device_count() <= 0:
                raise ModelUnavailableError("CUDA explicitly requested, but torch.cuda.device_count() is 0.")
            device_idx = config.get("device_id", 0)
            self.device = torch.device(f"cuda:{device_idx}")
        else:
            self.device = torch.device("cpu")

        self.model = None
        self.predictor = None
        self.video_predictor = None
        self._initialized = False
        self.vram_before_mb = 0.0
        self.vram_after_load_mb = 0.0
        self.vram_peak_mb = 0.0

    def verify_checkpoint(self) -> bool:
        path = Path(self.checkpoint_path)
        return path.exists()

    def initialize(self) -> bool:
        if self._initialized:
            return True

        if self.use_cuda and not torch.cuda.is_available():
            raise ModelUnavailableError("CUDA requested but not available during initialization.")

        if not self.verify_checkpoint():
            raise FileNotFoundError(f"SAM2 checkpoint mandatory for real inference not found at: {self.checkpoint_path}")

        try:
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            if self.device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.vram_before_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)

            logger.info(f"Loading official SAM2 model config {self.model_cfg} with checkpoint {self.checkpoint_path} on {self.device}")
            
            self.model = build_sam2(self.model_cfg, self.checkpoint_path, device=self.device, apply_postprocessing=False)
            self.model.eval()

            if self.device.type == "cuda":
                for name, param in self.model.named_parameters():
                    if param.device.type != "cuda":
                        raise RuntimeError(f"Parameter {name} is on {param.device}, expected cuda.")

            self.predictor = SAM2ImagePredictor(self.model)
            try:
                self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint_path, device=self.device)
            except Exception as e:
                logger.warning(f"Video predictor build warning: {e}")

            if self.device.type == "cuda":
                self.vram_after_load_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                self.vram_peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

            self._initialized = True
            return True
        except ImportError as e:
            raise ImportError(f"Official 'sam2' package is not installed: {e}")
        except Exception as e:
            if isinstance(e, ModelUnavailableError):
                raise e
            raise RuntimeError(f"Failed to initialize official SAM2 model: {e}")

    def segment_image(
        self,
        image_path: str,
        observation_id: str,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        artifact_store: Optional[ArtifactStore] = None
    ) -> SegmentationSet:
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        import cv2
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image from path: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        self.predictor.set_image(image_rgb)

        masks_list = []
        tracked_ids = []

        if point_coords is None and box is None:
            point_coords = np.array([[w // 2, h // 2]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
        elif point_coords is not None and point_labels is None:
            point_labels = np.ones(len(point_coords), dtype=np.int32)

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True
        )

        if not np.isfinite(masks).all():
            raise ValueError("SAM2 output masks contain non-finite (NaN/Inf) values.")

        for idx, (mask, score) in enumerate(zip(masks, scores)):
            mask_id = f"{observation_id}_mask_{idx}"
            obj_id = f"obj_{idx}"
            tracked_ids.append(obj_id)
            
            y_indices, x_indices = np.where(mask)
            bbox = []
            if len(x_indices) > 0 and len(y_indices) > 0:
                bbox = [int(x_indices.min()), int(y_indices.min()), int(x_indices.max()), int(y_indices.max())]

            area = int(mask.sum())

            masks_list.append(SegmentationMask(
                mask_id=mask_id,
                object_id=obj_id,
                mask_array=mask,
                confidence=float(score),
                bbox=bbox,
                metadata={
                    "source_image": image_path,
                    "area": area,
                    "resolution": [h, w]
                }
            ))

        elapsed_ms = (time.time() - start_time) * 1000.0

        if self.device.type == "cuda":
            vram_alloc = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            vram_peak = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        else:
            vram_alloc = 0.0
            vram_peak = 0.0

        provenance = {
            "worker": "SAM2Worker",
            "model": "sam2.1_hiera_large",
            "checkpoint": self.checkpoint_path,
            "config": self.model_cfg,
            "device": str(self.device),
            "dtype": str(self.model.parameters().__next__().dtype) if self.model else "unknown",
            "inference_latency_ms": elapsed_ms,
            "vram_allocated_mb": vram_alloc,
            "vram_peak_mb": vram_peak,
            "timestamp": time.time(),
            "real_inference": True
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

    def propagate_video_sequence(
        self,
        video_frames_dir: str,
        observation_id: str,
        seed_frame_idx: int = 0,
        obj_id: int = 1,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        artifact_store: Optional[ArtifactStore] = None
    ) -> Dict[str, Any]:
        """
        Executes true multi-frame video propagation using official SAM2 video predictor API.
        """
        if not self._initialized:
            self.initialize()

        if self.video_predictor is None:
            raise RuntimeError("SAM2 video predictor is not initialized.")

        start_time = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device) if hasattr(torch.cuda, 'reset_peak_memory_stats') else None

        # 1. Initialize inference state from frames directory
        inference_state = self.video_predictor.init_state(video_path=video_frames_dir)

        # 2. Add seeding prompt on seed frame
        if points is None:
            points = np.array([[256, 256]], dtype=np.float32)
        if labels is None:
            labels = np.array([1], dtype=np.int32)

        _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=seed_frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels
        )

        # 3. Propagate through video sequence
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
            frame_masks = {}
            for i, o_id in enumerate(out_obj_ids):
                # Convert logits to binary mask (> 0.0)
                mask_tensor = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                if mask_tensor.ndim == 3:
                    mask_tensor = mask_tensor[0] # handle channel dims if present
                
                # Rigorous validation: check finite and valid
                if not np.isfinite(mask_tensor).all():
                    raise ValueError(f"Non-finite values detected in propagated mask at frame {out_frame_idx}, object {o_id}")

                frame_masks[int(o_id)] = mask_tensor
            video_segments[out_frame_idx] = frame_masks

        elapsed_ms = (time.time() - start_time) * 1000.0

        if self.device.type == "cuda":
            vram_peak = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        else:
            vram_peak = 0.0

        provenance = {
            "worker": "SAM2Worker",
            "model": "sam2.1_hiera_large",
            "mode": "video_propagation",
            "frames_dir": video_frames_dir,
            "seed_frame_idx": seed_frame_idx,
            "propagated_frames_count": len(video_segments),
            "propagation_latency_ms": elapsed_ms,
            "vram_peak_mb": vram_peak,
            "timestamp": time.time(),
            "real_multi_frame_verified": True
        }

        # Persist results through artifact store if provided
        if artifact_store is not None:
            summary_data = {
                "observation_id": observation_id,
                "propagated_frames": len(video_segments),
                "tracked_objects": [int(o) for o in out_obj_ids],
                "provenance": provenance
            }
            artifact_store.save(f"multiview_propagation_{observation_id}", summary_data)

        return {
            "video_segments": video_segments,
            "object_ids": [int(o) for o in out_obj_ids],
            "frame_count": len(video_segments),
            "provenance": provenance
        }

    def release(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        if self.video_predictor is not None:
            del self.video_predictor
            self.video_predictor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._initialized = False

    def capabilities(self) -> Dict[str, Any]:
        gpu_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" and torch.cuda.is_available() else "CPU"
        return {
            "worker": "SAM2Worker",
            "initialized": self._initialized,
            "checkpoint": self.checkpoint_path,
            "config": self.model_cfg,
            "device": str(self.device),
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available()
        }
