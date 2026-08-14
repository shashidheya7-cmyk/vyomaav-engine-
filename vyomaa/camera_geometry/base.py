"""Abstract base class for geometry and Structure-from-Motion (SfM) adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..core.contracts import Camera, Observation
from ..core.exceptions import CameraGeometryError, ModelUnavailableError
from ..core.registry import ModelAdapter, ModelSpec


class BaseGeometryAdapter(ModelAdapter, ABC):
    """Abstract interface for multi-view geometry, point map prediction, and camera solvers."""

    def __init__(self, spec: ModelSpec, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(spec, config)
