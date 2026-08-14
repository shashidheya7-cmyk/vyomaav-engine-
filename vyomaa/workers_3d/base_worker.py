"""Base 3D worker adapter contract for generative shape workers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np

from ..core.contracts import ReconstructionHypothesis
from ..core.exceptions import GeometryError
from ..core.registry import ModelAdapter, ModelSpec
from ..core.types import ArtifactType, ModelCapability, ModelRuntimeState, PrecisionType
from ..representations.mesh import MeshData


class Base3DWorker(ModelAdapter):
    """Specialized worker for candidate 3D shape hypothesis generation."""

    def __init__(self, spec: ModelSpec, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(spec, config)

    @abstractmethod
    def generate_hypothesis(self, input_artifact: Any) -> Tuple[MeshData, ReconstructionHypothesis]:
        """Generate a 3D candidate mesh and associated hypothesis metadata."""
        raise NotImplementedError
