"""Integration tests running end-to-end Phase 2 DAG pipelines on real test data."""

import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np
from PIL import Image

from vyomaa.core.config import EngineConfig
from vyomaa.core.contracts import ValidationReport
from vyomaa.representations.point_cloud import PointCloud
from vyomaa.core.types import ModalityType, TaskState
from vyomaa.hardware.vram_manager import VRAMManager
from vyomaa.pipeline.dag import ExecutionPlan
from vyomaa.pipeline.task import TaskContext
from vyomaa.pipeline.tasks import (
    IngestionTask,
    VisionPerceptionTask,
    MultiViewEvidenceTask,
    PointMapBackprojectionTask,
    GeometricValidationTask,
)
from vyomaa.storage.artifact_store import ArtifactStore


class TestIntegrationPhase2(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.store = ArtifactStore(os.path.join(self.test_dir, "workspace"))
        self.vram_mgr = VRAMManager()
        self.config = EngineConfig()

        # 1. Real single image with geometric texture
        self.single_img = os.path.join(self.test_dir, "single.png")
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(arr, (100, 100), 50, (255, 200, 100), -1)
        cv2.rectangle(arr, (30, 30), (70, 70), (100, 255, 100), -1)
        cv2.imwrite(self.single_img, arr)

        # 2. Real multi-view set
        self.mv_dir = os.path.join(self.test_dir, "multiview")
        os.makedirs(self.mv_dir, exist_ok=True)
        for i in range(3):
            mv_arr = arr.copy()
            # Simulate slight perspective/affine shift
            M = np.float32([[1, 0, (i - 1) * 8], [0, 1, (i - 1) * 4]])
            shifted = cv2.warpAffine(mv_arr, M, (200, 200))
            cv2.imwrite(os.path.join(self.mv_dir, f"view_{i:02d}.png"), shifted)

        # 3. Real short video clip (15 frames)
        self.video_path = os.path.join(self.test_dir, "short_clip.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.video_path, fourcc, 15.0, (200, 200))
        for f in range(15):
            f_arr = arr.copy()
            cv2.circle(f_arr, (50 + f * 5, 100), 20, (255, 255, 255), -1)
            out.write(f_arr)
        out.release()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_single_image_pipeline(self):
        context = TaskContext(
            config=self.config,
            artifact_store=self.store,
            vram_manager=self.vram_mgr,
            diagnostics={"input_path": self.single_img, "modality": ModalityType.RGB_IMAGE},
        )
        plan = ExecutionPlan()
        plan.add_task(IngestionTask())
        plan.add_task(VisionPerceptionTask())
        plan.add_task(MultiViewEvidenceTask())
        plan.add_task(PointMapBackprojectionTask())
        plan.add_task(GeometricValidationTask())

        results = plan.execute(context)
        for tid, res in results.items():
            self.assertEqual(res.state, TaskState.COMPLETED, f"Task {tid} failed: {res.error_message}")

        # Check PointCloud artifact created
        pc_id = context.diagnostics.get("primary_pointcloud_id")
        self.assertIsNotNone(pc_id)
        pc = context.get_artifact(pc_id)
        self.assertIsInstance(pc, PointCloud)
        self.assertGreater(pc.point_count, 100)

        # Check Validation Report
        rep_id = context.diagnostics.get("validation_report_id")
        rep = context.get_artifact(rep_id)
        self.assertIsInstance(rep, ValidationReport)
        self.assertTrue(rep.is_valid)

    def test_multiview_pipeline(self):
        context = TaskContext(
            config=self.config,
            artifact_store=self.store,
            vram_manager=self.vram_mgr,
            diagnostics={"input_path": self.mv_dir, "modality": ModalityType.MULTIVIEW_IMAGE_SET},
        )
        plan = ExecutionPlan()
        plan.add_task(IngestionTask())
        plan.add_task(VisionPerceptionTask())
        plan.add_task(MultiViewEvidenceTask())
        plan.add_task(PointMapBackprojectionTask())
        plan.add_task(GeometricValidationTask())

        results = plan.execute(context)
        for tid, res in results.items():
            self.assertEqual(res.state, TaskState.COMPLETED, f"Task {tid} failed: {res.error_message}")

        # Verify multi-view view graph was evaluated
        vg = context.diagnostics.get("view_graph")
        self.assertIsNotNone(vg)
        self.assertEqual(len(vg.views), 3)

    def test_video_pipeline(self):
        context = TaskContext(
            config=self.config,
            artifact_store=self.store,
            vram_manager=self.vram_mgr,
            diagnostics={"input_path": self.video_path, "modality": ModalityType.MONOCULAR_VIDEO},
        )
        plan = ExecutionPlan()
        plan.add_task(IngestionTask())
        plan.add_task(VisionPerceptionTask())
        plan.add_task(MultiViewEvidenceTask())
        plan.add_task(PointMapBackprojectionTask())
        plan.add_task(GeometricValidationTask())

        results = plan.execute(context)
        self.assertEqual(results["task_ingest"].state, TaskState.COMPLETED)
        self.assertEqual(results["task_vision"].state, TaskState.COMPLETED)


if __name__ == "__main__":
    unittest.main()
