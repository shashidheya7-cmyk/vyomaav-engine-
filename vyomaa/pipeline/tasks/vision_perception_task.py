"""Vision Perception DAG task executing depth estimation, normals, and segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from ...core.contracts import DepthMap, MediaSequence, Observation
from ...core.exceptions import ModelUnavailableError
from ...core.provenance import ProvenanceRecord
from ...core.types import TaskState
from ...vision.bytetrack_tracker import ByteTrackTracker
from ...vision.depth_anything import DepthAnythingV2Adapter
from ...vision.surface_normals import SurfaceNormalEstimator
from ..task import Task, TaskContext, TaskDependency, TaskResult


class VisionPerceptionTask(Task):
    """
    Extract dense depth, surface normals, and tracking information.

    The critical contract is:

        observation_artifact_id
            ->
        diagnostics["depth_arrays"][observation_artifact_id]
            ->
        PointMapBackprojectionTask

    Neural depth and deterministic fallback depth therefore expose
    the same downstream representation.
    """

    def __init__(
        self,
        task_id: str = "task_vision",
        deps: Optional[List[TaskDependency]] = None,
        priority: int = 2,
    ) -> None:
        deps = deps or [TaskDependency("task_ingest")]

        super().__init__(
            task_id=task_id,
            name="Neural Perception & Depth Estimation",
            dependencies=deps,
            priority=priority,
        )

    def run(self, context: TaskContext) -> TaskResult:
        obs_ids = context.diagnostics.get(
            "observation_ids",
            [],
        )

        media_seq_id = context.diagnostics.get(
            "media_sequence_id"
        )

        depth_ids: List[str] = []

        depth_arrays: Dict[str, np.ndarray] = context.diagnostics.setdefault(
            "depth_arrays",
            {},
        )

        normal_arrays: Dict[str, np.ndarray] = context.diagnostics.setdefault(
            "normal_arrays",
            {},
        )

        for oid in obs_ids:
            obs = context.get_artifact(oid)

            if not isinstance(obs, Observation):
                continue

            if not obs.image_uri:
                continue

            img_bgr = cv2.imread(obs.image_uri)

            if img_bgr is None:
                continue

            height, width = img_bgr.shape[:2]

            # --------------------------------------------------------------
            # Depth
            # --------------------------------------------------------------

            depth_arr: Optional[np.ndarray] = None

            try:
                adapter = DepthAnythingV2Adapter(
                    context.config.perception.__dict__
                )

                adapter.initialize(
                    context.config.hardware.device,
                    context.config.hardware.precision,
                )

                img_pil = Image.fromarray(
                    cv2.cvtColor(
                        img_bgr,
                        cv2.COLOR_BGR2RGB,
                    )
                )

                depth_artifact = adapter.estimate_depth(
                    img_pil,
                    obs,
                )

                # IMPORTANT:
                # The real neural dense prediction must be handed to the
                # geometric stage. Previously only the DepthMap metadata
                # artifact was stored, leaving depth_arrays empty.
                depth_arr = adapter.last_depth_array

                if depth_arr is None:
                    raise ModelUnavailableError(
                        "DepthAnythingV2 returned a DepthMap without "
                        "a dense depth array."
                    )

                depth_arr = np.asarray(
                    depth_arr,
                    dtype=np.float32,
                )

                if depth_arr.shape != (height, width):
                    raise ModelUnavailableError(
                        "DepthAnythingV2 depth shape mismatch: "
                        f"{depth_arr.shape} != {(height, width)}"
                    )

                if not np.all(np.isfinite(depth_arr)):
                    raise ModelUnavailableError(
                        "DepthAnythingV2 returned non-finite depth values."
                    )

                context.put_artifact(depth_artifact)

                depth_ids.append(
                    depth_artifact.artifact_id
                )

            except ModelUnavailableError:
                # Deterministic fallback is retained intentionally so the
                # canonical software pipeline remains executable when the
                # neural worker is unavailable.
                gray = cv2.cvtColor(
                    img_bgr,
                    cv2.COLOR_BGR2GRAY,
                )

                norm_gray = (
                    gray.astype(np.float32) / 255.0
                )

                depth_arr = (
                    1.0 +
                    (1.0 - norm_gray) * 2.0
                )

                depth_artifact = DepthMap(
                    name=f"Depth_Prior_{obs.frame_id}",
                    width=width,
                    height=height,
                    min_depth=float(np.min(depth_arr)),
                    max_depth=float(np.max(depth_arr)),
                    is_metric=False,
                    camera_id=(
                        obs.camera.artifact_id
                        if obs.camera
                        else None
                    ),
                    confidence_score=0.70,
                    provenance=ProvenanceRecord(
                        producer_subsystem="vision",
                        producer_model="MonocularLuminancePrior",
                        parent_artifact_ids=[
                            obs.artifact_id
                        ],
                    ),
                )

                context.put_artifact(
                    depth_artifact
                )

                depth_ids.append(
                    depth_artifact.artifact_id
                )

            # --------------------------------------------------------------
            # Common dense-data handoff
            # --------------------------------------------------------------

            if depth_arr is None:
                continue

            depth_arr = np.asarray(
                depth_arr,
                dtype=np.float32,
            )

            depth_arrays[obs.artifact_id] = depth_arr

            # --------------------------------------------------------------
            # Surface normals
            # --------------------------------------------------------------

            normals_arr, _ = (
                SurfaceNormalEstimator.compute_normals_from_depth(
                    depth_arr,
                    obs.camera,
                )
            )

            normal_arrays[obs.artifact_id] = (
                np.asarray(
                    normals_arr,
                    dtype=np.float32,
                )
            )

        # --------------------------------------------------------------
        # Video
        # --------------------------------------------------------------

        if media_seq_id:
            media_seq = context.get_artifact(
                media_seq_id
            )

            if isinstance(
                media_seq,
                MediaSequence,
            ):
                tracker = ByteTrackTracker()
                tracker.initialize()

                frame_depth_arrays = context.diagnostics.setdefault(
                    "frame_depth_arrays",
                    {},
                )

                for frame in media_seq.frames:
                    if not frame.image_path:
                        continue

                    if not Path(frame.image_path).is_file():
                        continue

                    img_bgr = cv2.imread(
                        frame.image_path
                    )

                    if img_bgr is None:
                        continue

                    height, width = img_bgr.shape[:2]

                    norm_gray = (
                        cv2.cvtColor(
                            img_bgr,
                            cv2.COLOR_BGR2GRAY,
                        ).astype(np.float32)
                        / 255.0
                    )

                    depth_arr = (
                        1.0 +
                        (1.0 - norm_gray) * 2.0
                    )

                    depth_artifact = DepthMap(
                        name=f"Depth_{frame.name}",
                        width=width,
                        height=height,
                        min_depth=float(
                            np.min(depth_arr)
                        ),
                        max_depth=float(
                            np.max(depth_arr)
                        ),
                        confidence_score=0.70,
                        provenance=ProvenanceRecord(
                            producer_subsystem="vision",
                            parent_artifact_ids=[
                                frame.artifact_id
                            ],
                        ),
                    )

                    context.put_artifact(
                        depth_artifact
                    )

                    depth_ids.append(
                        depth_artifact.artifact_id
                    )

                    frame_depth_arrays[
                        frame.artifact_id
                    ] = depth_arr

        context.diagnostics[
            "depth_artifact_ids"
        ] = depth_ids

        context.diagnostics[
            "depth_array_count"
        ] = len(depth_arrays)

        context.diagnostics[
            "normal_array_count"
        ] = len(normal_arrays)

        return TaskResult(
            task_id=self.task_id,
            state=TaskState.COMPLETED,
            output_artifact_ids=depth_ids,
            diagnostics={
                "depth_maps_generated": len(depth_ids),
                "dense_depth_arrays": len(depth_arrays),
                "dense_normal_arrays": len(normal_arrays),
            },
        )
