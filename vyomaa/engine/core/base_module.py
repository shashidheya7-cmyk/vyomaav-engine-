
"""Common resource management utilities for engine modules."""

from abc import ABC
import gc
from .torch_compat import torch


class BaseModule(ABC):
    """Base module with deterministic host and CUDA-cache cleanup."""

    def cleanup(self) -> None:
        """Release collectable resources and cached CUDA allocations."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


