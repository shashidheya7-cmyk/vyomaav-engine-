"""VRAM budget manager and model residency coordinator."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional
import os

from ..core.exceptions import HardwareError
from .gpu_info import GPUInfo


@dataclass
class AllocationRecord:
    """Tracks memory assigned to an active or resident model worker."""

    model_name: str
    allocated_bytes: int
    residency_priority: int = 5  # 1 (low) to 10 (high)
    is_resident: bool = True
    tag: str = "general"

    @property
    def allocated_gb(self) -> float:
        return round(self.allocated_bytes / (1024 ** 3), 2)


class VRAMManager:
    """Manages high-VRAM allocations and model residency policies without destructive cache thrashing."""

    def __init__(self, reserve_headroom_gb: float = 8.0, max_vram_budget_gb: Optional[float] = None) -> None:
        self.lock = RLock()
        self.reserve_headroom_bytes = int(reserve_headroom_gb * (1024 ** 3))
        self.gpu_info = GPUInfo.discover()
        self.max_budget_bytes = (
            int(max_vram_budget_gb * (1024 ** 3))
            if max_vram_budget_gb
            else max(self.gpu_info.total_vram_bytes - self.reserve_headroom_bytes, 0)
        )
        self._allocations: Dict[str, AllocationRecord] = {}

    def get_status(self) -> Dict[str, Any]:
        """Return instantaneous memory allocation overview."""
        with self.lock:
            total_allocated = sum(a.allocated_bytes for a in self._allocations.values())
            return {
                "gpu_name": self.gpu_info.device_name,
                "total_vram_gb": self.gpu_info.total_vram_gb,
                "budget_limit_gb": round(self.max_budget_bytes / (1024 ** 3), 2),
                "managed_allocated_gb": round(total_allocated / (1024 ** 3), 2),
                "active_models": list(self._allocations.keys()),
            }

    def can_allocate(self, estimated_bytes: int) -> bool:
        """Check if requested memory fits within the managed VRAM budget."""
        with self.lock:
            current_allocated = sum(a.allocated_bytes for a in self._allocations.values())
            return (current_allocated + estimated_bytes) <= self.max_budget_bytes

    def register_allocation(self, model_name: str, estimated_bytes: int, priority: int = 5) -> AllocationRecord:
        """Track model worker memory footprint in residency pool."""
        with self.lock:
            if not self.can_allocate(estimated_bytes):
                raise HardwareError(
                    f"Requested {estimated_bytes / (1024**3):.2f} GB for '{model_name}' exceeds "
                    f"managed VRAM budget ({self.max_budget_bytes / (1024**3):.2f} GB)"
                )
            record = AllocationRecord(
                model_name=model_name,
                allocated_bytes=estimated_bytes,
                residency_priority=priority,
                is_resident=True,
            )
            self._allocations[model_name] = record
            return record

    def release_allocation(self, model_name: str) -> None:
        """Release allocation record without forcing premature host GC."""
        with self.lock:
            self._allocations.pop(model_name, None)
