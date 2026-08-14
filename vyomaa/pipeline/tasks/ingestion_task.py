"""Ingestion DAG task for images, multi-view collections, and video sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.contracts import InputArtifact, Observation
from ...core.types import ArtifactType, ModalityType, TaskState
from ...ingestion.image_loader import ImageLoader
from ...ingestion.rgbd_loader import RGBDLoader
from ...ingestion.video_processor import VideoProcessor
from ..task import Task, TaskContext, TaskDependency, TaskResult


class IngestionTask(Task):
    """Dispatches media ingestion according to configured modality."""

    def __init__(self, task_id: str = "task_ingest", priority: int = 1) -> None:
        super().__init__(task_id=task_id, name="Media Ingestion & Invariant Extraction", priority=priority)

    def run(self, context: TaskContext) -> TaskResult:
        input_path = context.diagnostics.get("input_path")
        modality = context.diagnostics.get("modality", ModalityType.RGB_IMAGE)

        if not input_path:
            return TaskResult(task_id=self.task_id, state=TaskState.FAILED, error_message="No input_path provided in context diagnostics")

        p = Path(input_path)
        output_ids: List[str] = []

        if modality == ModalityType.RGB_IMAGE:
            input_art, obs = ImageLoader.load_single_image(str(p))
            context.put_artifact(input_art)
            context.put_artifact(obs)
            context.diagnostics["primary_input_id"] = input_art.artifact_id
            context.diagnostics["observation_ids"] = [obs.artifact_id]
            output_ids.extend([input_art.artifact_id, obs.artifact_id])

        elif modality == ModalityType.MULTIVIEW_IMAGE_SET:
            input_art, obs_list = ImageLoader.load_multiview_directory(str(p))
            context.put_artifact(input_art)
            obs_ids = []
            for o in obs_list:
                context.put_artifact(o)
                obs_ids.append(o.artifact_id)
            context.diagnostics["primary_input_id"] = input_art.artifact_id
            context.diagnostics["observation_ids"] = obs_ids
            output_ids.append(input_art.artifact_id)
            output_ids.extend(obs_ids)

        elif modality == ModalityType.MONOCULAR_VIDEO:
            frames_dir = context.artifact_store.artifacts_dir / "observations" / f"{p.stem}_frames"
            input_art, media_seq = VideoProcessor.ingest_video(
                str(p),
                output_frames_dir=str(frames_dir),
                keyframe_stride=context.config.ingestion.max_video_frames // 30 or 5,
                max_keyframes=50,
            )
            context.put_artifact(input_art)
            context.put_artifact(media_seq)
            context.diagnostics["primary_input_id"] = input_art.artifact_id
            context.diagnostics["media_sequence_id"] = media_seq.artifact_id
            output_ids.extend([input_art.artifact_id, media_seq.artifact_id])

        return TaskResult(
            task_id=self.task_id,
            state=TaskState.COMPLETED,
            output_artifact_ids=output_ids,
            diagnostics={"modality": modality.value if isinstance(modality, ModalityType) else str(modality)},
        )
