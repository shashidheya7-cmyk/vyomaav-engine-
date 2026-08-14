"""Hardware introspection and GPU resource discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os


@dataclass
class GPUInfo:
    """Introspected GPU device properties and real-time VRAM telemetry."""

    is_cuda_available: bool = False
    device_count: int = 0
    device_name: str = "CPU Only"
    total_vram_bytes: int = 0
    free_vram_bytes: int = 0
    compute_capability: Tuple[int, int] = (0, 0)
    is_blackwell_class: bool = False

    @property
    def total_vram_gb(self) -> float:
        return round(self.total_vram_bytes / (1024 ** 3), 2)

    @property
    def free_vram_gb(self) -> float:
        return round(self.free_vram_bytes / (1024 ** 3), 2)

    @classmethod
    def discover(cls, device_index: int = 0) -> GPUInfo:
        """Query PyTorch CUDA subsystem for current hardware state."""
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                idx = min(device_index, torch.cuda.device_count() - 1)
                props = torch.cuda.get_device_properties(idx)
                total = props.total_memory
                # Query real-time memory
                free, _ = torch.cuda.mem_get_info(idx)
                cap = (props.major, props.minor)
                is_blackwell = (props.major >= 10) or ("Blackwell" in props.name) or ("RTX PRO 6000" in props.name and total > 80 * (1024**3))
                return cls(
                    is_cuda_available=True,
                    device_count=torch.cuda.device_count(),
                    device_name=props.name,
                    total_vram_bytes=total,
                    free_vram_bytes=free,
                    compute_capability=cap,
                    is_blackwell_class=is_blackwell,
                )
        except Exception:
            pass

        return cls(
            is_cuda_available=False,
            device_count=0,
            device_name="CPU Only",
            total_vram_bytes=0,
            free_vram_bytes=0,
            compute_capability=(0, 0),
            is_blackwell_class=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_cuda_available": self.is_cuda_available,
            "device_count": self.device_count,
            "device_name": self.device_name,
            "total_vram_gb": self.total_vram_gb,
            "free_vram_gb": self.free_vram_gb,
            "compute_capability": list(self.compute_capability),
            "is_blackwell_class": self.is_blackwell_class,
        }
