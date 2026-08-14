"""Directed Acyclic Graph (DAG) planner and execution coordinator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from ..core.exceptions import PipelineExecutionError, TaskDependencyError
from ..core.types import TaskState
from .task import Task, TaskContext, TaskResult


class ExecutionPlan:
    """Deterministic DAG execution planner resolving task dependency orders."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        """Add task to the execution graph."""
        if task.task_id in self._tasks:
            raise TaskDependencyError(f"Task with ID '{task.task_id}' is already present in ExecutionPlan")
        self._tasks[task.task_id] = task

    def get_topological_order(self) -> List[Task]:
        """Compute valid topological task ordering or raise if cycle detected (Kahn's algorithm)."""
        in_degree: Dict[str, int] = {t_id: 0 for t_id in self._tasks}
        graph: Dict[str, List[str]] = {t_id: [] for t_id in self._tasks}

        for t_id, task in self._tasks.items():
            for dep in task.dependencies:
                if dep.parent_task_id not in self._tasks:
                    if not dep.is_optional:
                        raise TaskDependencyError(
                            f"Task '{t_id}' depends on non-existent task '{dep.parent_task_id}'"
                        )
                    continue
                graph[dep.parent_task_id].append(t_id)
                in_degree[t_id] += 1

        # Queue nodes with 0 incoming dependencies
        zero_in = [t_id for t_id, deg in in_degree.items() if deg == 0]
        # Sort by priority
        zero_in.sort(key=lambda tid: self._tasks[tid].priority)

        ordered: List[Task] = []
        while zero_in:
            curr_id = zero_in.pop(0)
            ordered.append(self._tasks[curr_id])

            for child_id in graph[curr_id]:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    zero_in.append(child_id)
                    zero_in.sort(key=lambda tid: self._tasks[tid].priority)

        if len(ordered) != len(self._tasks):
            unresolved = set(self._tasks.keys()) - {t.task_id for t in ordered}
            raise TaskDependencyError(f"Circular dependency cycle detected among tasks: {unresolved}")

        return ordered

    def execute(self, context: TaskContext) -> Dict[str, TaskResult]:
        """Execute the task DAG in deterministic topological sequence."""
        ordered_tasks = self.get_topological_order()
        results: Dict[str, TaskResult] = {}

        for task in ordered_tasks:
            # Check if mandatory parent tasks succeeded
            can_run = True
            for dep in task.dependencies:
                if dep.parent_task_id in results:
                    parent_res = results[dep.parent_task_id]
                    if parent_res.state == TaskState.FAILED and not dep.is_optional:
                        can_run = False
                        break

            if not can_run:
                task.state = TaskState.SKIPPED
                results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    state=TaskState.SKIPPED,
                    error_message="Skipped due to upstream parent task failure.",
                )
                continue

            result = task.execute(context)
            results[task.task_id] = result
            if result.state == TaskState.FAILED:
                # Task failure recorded
                pass

        return results
