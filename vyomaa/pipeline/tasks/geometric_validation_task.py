"""Geometric Validation DAG task performing multi-factor quality audits."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...core.contracts import Camera, DepthMap, Observation, ValidationReport
from ...core.types import TaskState
from ...multiview.view_graph import ViewGraph
from ...representations.point_cloud import PointCloud
from ...validation.geometric_validator import GeometricValidator
from ..task import Task, TaskContext, TaskDependency, TaskResult


class GeometricValidationTask(Task):
    """Executes geometric validation and produces the Phase 2 ValidationReport."""

    def __init__(self, task_id: str = "task_validation", deps: Optional[List[TaskDependency]] = None, priority: int = 5) -> None:
        deps = deps or [TaskDependency("task_pointmap")]
        super().__init__(task_id=task_id, name="Geometric Evidence Validation & Quality Scoring", dependencies=deps, priority=priority)

    def run(self, context: TaskContext) -> TaskResult:
        obs_ids = context.diagnostics.get("observation_ids", [])
        depth_ids = context.diagnostics.get("depth_artifact_ids", [])
        pc_id = context.diagnostics.get("primary_pointcloud_id")
        view_graph = context.diagnostics.get("view_graph")

        depth_maps: List[DepthMap] = []
        cameras: List[Camera] = []

        for did in depth_ids:
            art = context.get_artifact(did)
            if isinstance(art, DepthMap):
                depth_maps.append(art)

        for oid in obs_ids:
            obs = context.get_artifact(oid)
            if isinstance(obs, Observation) and obs.camera:
                cameras.append(obs.camera)

        point_cloud = context.get_artifact(pc_id) if pc_id else None
        if not isinstance(point_cloud, PointCloud):
            point_cloud = None

        report = GeometricValidator.generate_comprehensive_report(
            depth_maps=depth_maps,
            cameras=cameras,
            point_cloud=point_cloud,
            view_graph=view_graph if isinstance(view_graph, ViewGraph) else None,
        )

        context.put_artifact(report)
        context.diagnostics["validation_report_id"] = report.artifact_id

        return TaskResult(
            task_id=self.task_id,
            state=TaskState.COMPLETED,
            output_artifact_ids=[report.artifact_id],
            diagnostics={
                "overall_quality_score": report.overall_quality_score,
                "is_valid": report.is_valid,
                "warnings_count": len(report.warnings),
            },
        )
