"""Multi-view Evidence DAG task performing feature correspondence and epipolar verification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from ...core.contracts import Camera, Observation
from ...core.types import TaskState
from ...multiview.correspondence import CorrespondenceEngine
from ...multiview.epipolar_checker import EpipolarChecker
from ...multiview.view_graph import ViewGraph, ViewPair
from ...multiview.view_selector import ViewSelector
from ..task import Task, TaskContext, TaskDependency, TaskResult


class MultiViewEvidenceTask(Task):
    """Builds ViewGraph, verifies epipolar consistency, and selects informative keyframes."""

    def __init__(self, task_id: str = "task_multiview", deps: Optional[List[TaskDependency]] = None, priority: int = 3) -> None:
        deps = deps or [TaskDependency("task_ingest")]
        super().__init__(task_id=task_id, name="Multi-View Evidence & Epipolar Geometry", dependencies=deps, priority=priority)

    def run(self, context: TaskContext) -> TaskResult:
        obs_ids = context.diagnostics.get("observation_ids", [])
        if len(obs_ids) < 2:
            return TaskResult(
                task_id=self.task_id,
                state=TaskState.COMPLETED,
                diagnostics={"note": "Single observation mode; multi-view correspondence bypassed."},
            )

        corr_engine = CorrespondenceEngine(detector_type="SIFT", max_features=2000)
        view_graph = ViewGraph()

        # Load observations and images
        obs_list: List[Observation] = []
        img_list: List[np.ndarray] = []

        for oid in obs_ids:
            obs = context.get_artifact(oid)
            if isinstance(obs, Observation) and obs.image_uri:
                img = cv2.imread(obs.image_uri)
                if img is not None:
                    obs_list.append(obs)
                    img_list.append(img)
                    q_score = corr_engine.evaluate_quality(obs.artifact_id, img)
                    view_graph.add_view(obs, q_score)

        # Match all pairs
        valid_pairs: List[ViewPair] = []
        for i in range(len(obs_list)):
            for j in range(i + 1, len(obs_list)):
                corr_map = corr_engine.match_views(obs_list[i], img_list[i], obs_list[j], img_list[j])
                pair = EpipolarChecker.validate_pair(obs_list[i], obs_list[j], corr_map)
                view_graph.add_edge(pair)
                if pair.is_valid_edge:
                    valid_pairs.append(pair)

        # Select non-redundant views
        selected_view_ids = ViewSelector.select_informative_views(view_graph)
        context.diagnostics["view_graph"] = view_graph
        context.diagnostics["selected_view_ids"] = selected_view_ids

        return TaskResult(
            task_id=self.task_id,
            state=TaskState.COMPLETED,
            diagnostics={
                "total_views": len(obs_list),
                "total_edges": len(view_graph.edges),
                "valid_edges": len(valid_pairs),
                "selected_views": len(selected_view_ids),
            },
        )
