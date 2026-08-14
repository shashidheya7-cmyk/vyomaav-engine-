"""RGB-D paired image and depth sequence loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image

from ..core.contracts import DepthMap, InputArtifact, Observation, Camera
from ..core.exceptions import IngestionError
from ..core.provenance import ProvenanceRecord
from ..core.types import ArtifactType, ModalityType


class RGBDLoader:
    """Loads paired RGB images and metric/disparity depth arrays."""

    @staticmethod
    def load_rgbd_pair(
        rgb_path: str,
        depth_path: str,
        depth_scale: float = 1000.0,  # Millimeters to meters
        is_metric: bool = True,
    ) -> Tuple[InputArtifact, Observation, DepthMap]:
        """Load RGB image and corresponding 16-bit or floating-point depth map."""
        p_rgb, p_depth = Path(rgb_path).resolve(), Path(depth_path).resolve()
        if not p_rgb.is_file():
            raise IngestionError(f"RGB file not found: {rgb_path}")
        if not p_depth.is_file():
            raise IngestionError(f"Depth file not found: {depth_path}")

        try:
            with Image.open(p_rgb) as img:
                w, h = img.size
        except Exception as exc:
            raise IngestionError(f"Failed to read RGB image: {exc}") from exc

        # Read depth array
        try:
            if p_depth.suffix.lower() == ".npy":
                depth_arr = np.load(str(p_depth)).astype(np.float32)
            else:
                with Image.open(p_depth) as d_img:
                    depth_raw = np.array(d_img)
                    depth_arr = (depth_raw.astype(np.float32) / depth_scale) if depth_scale > 0 else depth_raw.astype(np.float32)
        except Exception as exc:
            raise IngestionError(f"Failed to decode depth map: {exc}") from exc

        min_d = float(np.min(depth_arr[depth_arr > 0])) if np.any(depth_arr > 0) else 0.0
        max_d = float(np.max(depth_arr))

        focal = float(max(w, h)) * 1.2
        camera = Camera.from_matrices(
            K=np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            RT=np.eye(4, dtype=np.float32),
            image_size=(w, h),
        )

        input_art = InputArtifact(
            name=f"RGBD_{p_rgb.stem}",
            modality=ModalityType.RGBD_IMAGE,
            file_path=str(p_rgb),
            resolution=(w, h),
            channels=4,
            provenance=ProvenanceRecord(producer_subsystem="ingestion"),
        )

        obs = Observation(
            name=f"Obs_RGBD_{p_rgb.stem}",
            frame_id=p_rgb.stem,
            camera=camera,
            image_uri=str(p_rgb),
            resolution=(w, h),
            is_primary_view=True,
            provenance=ProvenanceRecord(
                producer_subsystem="ingestion",
                parent_artifact_ids=[input_art.artifact_id],
            ),
        )

        depth_art = DepthMap(
            name=f"Depth_{p_depth.stem}",
            width=w,
            height=h,
            min_depth=min_d,
            max_depth=max_d,
            is_metric=is_metric,
            storage_path=str(p_depth),
            camera_id=camera.artifact_id,
            provenance=ProvenanceRecord(
                producer_subsystem="ingestion",
                parent_artifact_ids=[input_art.artifact_id],
            ),
        )

        return input_art, obs, depth_art
