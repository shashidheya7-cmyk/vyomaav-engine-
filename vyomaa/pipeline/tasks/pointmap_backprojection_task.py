"""World-Space PointMap DAG task unprojecting depth into canonical 3D PointClouds."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from ...core.contracts import Camera, DepthMap, Observation
from ...core.types import TaskState
from ...pointmap.backprojector import DepthBackprojector
from ...representations.point_cloud import PointCloud
from ..task import Task, TaskContext, TaskDependency, TaskResult


class PointMapBackprojectionTask(Task):
    """Converts 2D observations + dense depth into metric 3D point cloud artifacts."""

    def __init__(self, task_id: str = "task_pointmap", deps: Optional[List[TaskDependency]] = None, priority: int = 4) -> None:
        deps = deps or [TaskDependency("task_vision")]
        super().__init__(task_id=task_id, name="3D Back-Projection & Point Cloud Assembly", dependencies=deps, priority=priority)

    def run(self, context: TaskContext) -> TaskResult:
        obs_ids = context.diagnostics.get("observation_ids", [])
        depth_arrays = context.diagnostics.get("depth_arrays", {})
        normal_arrays = context.diagnostics.get("normal_arrays", {})

        generated_clouds: List[PointCloud] = []

        for oid in obs_ids:
            obs = context.get_artifact(oid)
            if not isinstance(obs, Observation) or oid not in depth_arrays:
                continue

            d_arr = depth_arrays[oid]
            n_arr = normal_arrays.get(oid)
            img_rgb = None
            if obs.image_uri:
                bgr = cv2.imread(obs.image_uri)
                if bgr is not None:
                    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            pc = DepthBackprojector.backproject_observation(
                observation=obs,
                depth_array=d_arr,
                image_rgb=img_rgb,
                normals_array=n_arr,
                stride=2,  # Subsample for dense point cloud
            )
            context.put_artifact(pc)
            generated_clouds.append(pc)

        # Merge point clouds if multi-view
        if len(generated_clouds) > 1:
            merged_pc = DepthBackprojector.merge_point_clouds(generated_clouds, voxel_size=0.01)
            context.put_artifact(merged_pc)
            context.diagnostics["primary_pointcloud_id"] = merged_pc.artifact_id
            out_ids = [c.artifact_id for c in generated_clouds] + [merged_pc.artifact_id]
        elif len(generated_clouds) == 1:
            context.diagnostics["primary_pointcloud_id"] = generated_clouds[0].artifact_id
            out_ids = [generated_clouds[0].artifact_id]
        else:
            out_ids = []

        return TaskResult(
            task_id=self.task_id,
            state=TaskState.COMPLETED,
            output_artifact_ids=out_ids,
            diagnostics={"point_clouds_created": len(generated_clouds)},
        )
