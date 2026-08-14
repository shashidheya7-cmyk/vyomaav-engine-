"""Task and context models for deterministic DAG pipeline execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from ..core.base_artifact import BaseArtifact
from ..core.config import EngineConfig
from ..core.types import ArtifactType, TaskState
from ..hardware.vram_manager import VRAMManager
from ..storage.artifact_store import ArtifactStore


@dataclass
class TaskDependency:
    """Explicit dependency requirement on prior task outputs."""

    parent_task_id: str
    required_artifact_type: Optional[ArtifactType] = None
    is_optional: bool = False


@dataclass
class TaskContext:
    """Execution context provided to tasks during pipeline runs."""

    config: EngineConfig
    artifact_store: ArtifactStore
    vram_manager: VRAMManager
    artifacts: Dict[str, BaseArtifact] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def get_artifact(self, artifact_id: str) -> Optional[BaseArtifact]:
        """Retrieve an in-memory or persisted artifact."""
        if artifact_id in self.artifacts:
            return self.artifacts[artifact_id]
        return None

    def put_artifact(self, artifact: BaseArtifact) -> None:
        """Register artifact in context and persist to artifact store."""
        self.artifacts[artifact.artifact_id] = artifact
        self.artifact_store.save_artifact(artifact)


@dataclass
class TaskResult:
    """Result summary produced by task execution."""

    task_id: str
    state: TaskState = TaskState.COMPLETED
    output_artifact_ids: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class Task(ABC):
    """Abstract unit of work in the VYOMAA DAG pipeline."""

    def __init__(
        self,
        task_id: str,
        name: str = "unnamed_task",
        dependencies: Optional[List[TaskDependency]] = None,
        priority: int = 5,
    ) -> None:
        self.task_id = task_id
        self.name = name
        self.dependencies = dependencies or []
        self.priority = priority
        self.state = TaskState.PENDING

    @abstractmethod
    def run(self, context: TaskContext) -> TaskResult:
        """Execute deterministic task logic."""
        raise NotImplementedError

    def execute(self, context: TaskContext) -> TaskResult:
        """Wrapped execution with timing and error isolation."""
        self.state = TaskState.RUNNING
        start = time.perf_counter()
        try:
            result = self.run(context)
            result.execution_time_seconds = round(time.perf_counter() - start, 4)
            self.state = result.state
            return result
        except Exception as exc:
            self.state = TaskState.FAILED
            return TaskResult(
                task_id=self.task_id,
                state=TaskState.FAILED,
                execution_time_seconds=round(time.perf_counter() - start, 4),
                error_message=str(exc),
            )
