"""Unit tests for DAG Task scheduling, Kahn topological sorting, and cycle detection."""

import shutil
import tempfile
import unittest

from vyomaa.core.config import EngineConfig
from vyomaa.core.exceptions import TaskDependencyError
from vyomaa.core.types import TaskState
from vyomaa.hardware.vram_manager import VRAMManager
from vyomaa.pipeline.dag import ExecutionPlan
from vyomaa.pipeline.task import Task, TaskContext, TaskDependency, TaskResult
from vyomaa.storage.artifact_store import ArtifactStore


class DummyTask(Task):
    def __init__(self, task_id: str, deps=None, priority=5, fail=False):
        super().__init__(task_id=task_id, dependencies=deps, priority=priority)
        self.fail = fail

    def run(self, context: TaskContext) -> TaskResult:
        if self.fail:
            raise RuntimeError(f"Task {self.task_id} simulated error")
        return TaskResult(task_id=self.task_id, state=TaskState.COMPLETED)


class TestPipelineDAG(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.context = TaskContext(
            config=EngineConfig(),
            artifact_store=ArtifactStore(self.test_dir),
            vram_manager=VRAMManager(),
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_deterministic_topological_execution(self):
        t1 = DummyTask("ingest", priority=1)
        t2 = DummyTask("perception", deps=[TaskDependency("ingest")], priority=2)
        t3 = DummyTask("geometry", deps=[TaskDependency("perception")], priority=3)

        plan = ExecutionPlan()
        plan.add_task(t3)
        plan.add_task(t1)
        plan.add_task(t2)

        ordered = plan.get_topological_order()
        self.assertEqual([t.task_id for t in ordered], ["ingest", "perception", "geometry"])

        results = plan.execute(self.context)
        self.assertEqual(results["ingest"].state, TaskState.COMPLETED)
        self.assertEqual(results["perception"].state, TaskState.COMPLETED)
        self.assertEqual(results["geometry"].state, TaskState.COMPLETED)

    def test_circular_dependency_detection(self):
        t1 = DummyTask("A", deps=[TaskDependency("B")])
        t2 = DummyTask("B", deps=[TaskDependency("A")])

        plan = ExecutionPlan()
        plan.add_task(t1)
        plan.add_task(t2)

        with self.assertRaises(TaskDependencyError):
            plan.get_topological_order()

    def test_upstream_failure_skips_downstream(self):
        t1 = DummyTask("step1", fail=True)
        t2 = DummyTask("step2", deps=[TaskDependency("step1")])

        plan = ExecutionPlan()
        plan.add_task(t1)
        plan.add_task(t2)

        results = plan.execute(self.context)
        self.assertEqual(results["step1"].state, TaskState.FAILED)
        self.assertEqual(results["step2"].state, TaskState.SKIPPED)


if __name__ == "__main__":
    unittest.main()
