"""Image ingestion loader for single RGB images and multi-view image collections."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from ..core.contracts import InputArtifact, Observation, Camera
from ..core.exceptions import IngestionError
from ..core.provenance import ProvenanceRecord
from ..core.types import ArtifactType, ModalityType
from .metadata_extractor import MetadataExtractor


class ImageLoader:
    """Loads and validates single RGB images or multi-view image sets into typed artifacts."""

    @staticmethod
    def load_single_image(image_path: str, name: str = "Primary Image") -> Tuple[InputArtifact, Observation]:
        """Load single RGB image, extract metadata, and produce InputArtifact & Observation."""
        p = Path(image_path).resolve()
        if not p.is_file():
            raise IngestionError(f"Input image not found: {image_path}")

        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                width, height = img_rgb.size
        except Exception as exc:
            raise IngestionError(f"Failed to decode image from {image_path}: {exc}") from exc

        exif = MetadataExtractor.extract_image_exif(str(p))

        # Default focal length estimation from FOV ~60 degrees if EXIF missing
        focal = float(max(width, height)) * 1.2
        if exif.get("focal_length_35mm"):
            focal = (float(exif["focal_length_35mm"]) / 36.0) * float(width)

        camera = Camera.from_matrices(
            K=np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            RT=np.eye(4, dtype=np.float32),
            image_size=(width, height),
            name=f"Camera_{p.stem}",
        )

        input_artifact = InputArtifact(
            name=name,
            modality=ModalityType.RGB_IMAGE,
            file_path=str(p),
            resolution=(width, height),
            channels=3,
            color_space="sRGB",
            provenance=ProvenanceRecord(
                producer_subsystem="ingestion",
                generation_parameters={"exif": exif},
            ),
        )

        observation = Observation(
            name=f"Obs_{p.stem}",
            frame_id=p.stem,
            camera=camera,
            image_uri=str(p),
            resolution=(width, height),
            is_primary_view=True,
            provenance=ProvenanceRecord(
                producer_subsystem="ingestion",
                parent_artifact_ids=[input_artifact.artifact_id],
            ),
        )

        return input_artifact, observation

    @staticmethod
    def load_multiview_directory(directory_path: str) -> Tuple[InputArtifact, List[Observation]]:
        """Load directory of multi-view RGB images in alphabetical/sorted order."""
        dir_p = Path(directory_path).resolve()
        if not dir_p.is_dir():
            raise IngestionError(f"Multi-view directory does not exist: {directory_path}")

        extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
        image_files = sorted([f for f in dir_p.iterdir() if f.is_file() and f.suffix.lower() in extensions])
        if not image_files:
            raise IngestionError(f"No valid image files found in multi-view directory: {directory_path}")

        observations: List[Observation] = []
        file_paths: List[str] = []
        first_res = (0, 0)

        for idx, f in enumerate(image_files):
            file_paths.append(str(f))
            try:
                with Image.open(f) as img:
                    w, h = img.size
                    if idx == 0:
                        first_res = (w, h)
            except Exception as exc:
                raise IngestionError(f"Failed to decode multi-view image {f}: {exc}") from exc

            focal = float(max(w, h)) * 1.2
            camera = Camera.from_matrices(
                K=np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                RT=np.eye(4, dtype=np.float32),
                image_size=(w, h),
                name=f"Camera_MV_{idx:03d}",
            )

            obs = Observation(
                name=f"Obs_MV_{idx:03d}",
                frame_id=f.stem,
                camera=camera,
                image_uri=str(f),
                resolution=(w, h),
                is_primary_view=(idx == 0),
                provenance=ProvenanceRecord(producer_subsystem="ingestion"),
            )
            observations.append(obs)

        input_artifact = InputArtifact(
            name=f"MultiView Set ({len(image_files)} views)",
            modality=ModalityType.MULTIVIEW_IMAGE_SET,
            file_paths=file_paths,
            resolution=first_res,
            channels=3,
            color_space="sRGB",
            provenance=ProvenanceRecord(
                producer_subsystem="ingestion",
                generation_parameters={"view_count": len(image_files)},
            ),
        )

        for obs in observations:
            obs.provenance.add_parent(input_artifact.artifact_id)

        return input_artifact, observations
