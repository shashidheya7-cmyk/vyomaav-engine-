"""Streaming and chunked video ingestion without full-RAM frame duplication."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import cv2
import numpy as np

from ..core.contracts import FrameArtifact, InputArtifact, MediaSequence
from ..core.exceptions import IngestionError
from ..core.provenance import ProvenanceRecord
from ..core.types import ModalityType


class VideoProcessor:
    """Processes monocular RGB video streams with frame streaming and keyframe indexing."""

    @staticmethod
    def get_video_metadata(video_path: str) -> Dict[str, Any]:
        """Extract frame count, FPS, duration, codec, and resolution."""
        p = Path(video_path).resolve()
        if not p.is_file():
            raise IngestionError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise IngestionError(f"Failed to open video container: {video_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = float(total_frames) / fps if fps > 0 else 0.0
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        finally:
            cap.release()

        return {
            "file_path": str(p),
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": round(duration, 3),
            "codec": codec,
        }

    @staticmethod
    def stream_frames(
        video_path: str,
        max_frames: Optional[int] = None,
        stride: int = 1,
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield (frame_index, timestamp_seconds, bgr_array) in streaming fashion."""
        p = Path(video_path).resolve()
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise IngestionError(f"Cannot open video stream: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_idx = 0
        yielded_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % stride == 0:
                    timestamp = float(frame_idx) / fps
                    yield (frame_idx, timestamp, frame)
                    yielded_count += 1
                    if max_frames is not None and yielded_count >= max_frames:
                        break
                frame_idx += 1
        finally:
            cap.release()

    @staticmethod
    def ingest_video(
        video_path: str,
        output_frames_dir: Optional[str] = None,
        keyframe_stride: int = 10,
        max_keyframes: int = 100,
    ) -> Tuple[InputArtifact, MediaSequence]:
        """Decode and catalog video keyframes into a canonical MediaSequence."""
        meta = VideoProcessor.get_video_metadata(video_path)
        p = Path(video_path).resolve()

        input_art = InputArtifact(
            name=f"Video_{p.stem}",
            modality=ModalityType.MONOCULAR_VIDEO,
            file_path=str(p),
            resolution=(meta["width"], meta["height"]),
            channels=3,
            fps=meta["fps"],
            duration_seconds=meta["duration_seconds"],
            provenance=ProvenanceRecord(producer_subsystem="ingestion", generation_parameters=meta),
        )

        frames_dir = Path(output_frames_dir) if output_frames_dir else p.parent / f"{p.stem}_extracted_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_artifacts: List[FrameArtifact] = []
        for idx, timestamp, bgr_frame in VideoProcessor.stream_frames(str(p), max_frames=max_keyframes, stride=keyframe_stride):
            frame_filename = f"frame_{idx:06d}.png"
            frame_path = frames_dir / frame_filename
            cv2.imwrite(str(frame_path), bgr_frame)

            fa = FrameArtifact(
                name=f"Frame_{idx:06d}",
                frame_index=idx,
                timestamp_seconds=timestamp,
                sequence_id=input_art.artifact_id,
                image_path=str(frame_path),
                resolution=(meta["width"], meta["height"]),
                is_keyframe=True,
                provenance=ProvenanceRecord(
                    producer_subsystem="ingestion",
                    parent_artifact_ids=[input_art.artifact_id],
                ),
            )
            frame_artifacts.append(fa)

        media_seq = MediaSequence(
            name=f"Sequence_{p.stem}",
            frames=frame_artifacts,
            total_frames=len(frame_artifacts),
            fps=meta["fps"],
            provenance=ProvenanceRecord(
                producer_subsystem="ingestion",
                parent_artifact_ids=[input_art.artifact_id],
            ),
        )

        return input_art, media_seq
