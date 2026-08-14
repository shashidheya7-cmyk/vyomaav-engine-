"""Abstract base adapter for neural vision perception models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..core.exceptions import VisionError, ModelUnavailableError
from ..core.registry import ModelAdapter, ModelSpec


class BaseVisionAdapter(ModelAdapter, ABC):
    """Base adapter interface for depth, segmentation, tracking, and normal networks."""

    def __init__(self, spec: ModelSpec, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(spec, config)
