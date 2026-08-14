"""Deterministic single-GPU task execution lock and priority scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import PriorityQueue
from threading import RLock
from typing import Any, Callable, Dict, Optional

from ..core.exceptions import HardwareError
from .vram_manager import VRAMManager


@dataclass(order=True)
class ScheduledItem:
    priority: int
    task_id: str = field(compare=False)
    executable_fn: Callable[..., Any] = field(compare=False)


class SafeScheduler:
    """Thread-safe sequential task scheduler designed for high-VRAM single GPU execution."""

    def __init__(self, vram_manager: Optional[VRAMManager] = None) -> None:
        self.vram_manager = vram_manager or VRAMManager()
        self._lock = RLock()
        self._queue: PriorityQueue[ScheduledItem] = PriorityQueue()

    def submit(self, task_id: str, priority: int, fn: Callable[..., Any]) -> None:
        """Enqueue task with priority (lower number = higher execution priority)."""
        self._queue.put(ScheduledItem(priority=priority, task_id=task_id, executable_fn=fn))

    def execute_next(self) -> Any:
        """Execute the next highest priority item in the queue with GPU exclusive lock."""
        with self._lock:
            if self._queue.empty():
                return None
            item = self._queue.get()
            return item.executable_fn()

    def execute_all(self) -> Dict[str, Any]:
        """Execute all queued items in deterministic priority order."""
        results = {}
        with self._lock:
            while not self._queue.empty():
                item = self._queue.get()
                results[item.task_id] = item.executable_fn()
        return results
