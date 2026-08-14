"""Hardware foundation package for device introspection, VRAM budgeting, and scheduling."""

from .gpu_info import GPUInfo
from .vram_manager import VRAMManager, AllocationRecord
from .scheduler import SafeScheduler

__all__ = [
    "GPUInfo",
    "VRAMManager",
    "AllocationRecord",
    "SafeScheduler",
]
