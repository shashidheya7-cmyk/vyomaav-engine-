"""TripoSR worker adapter for rapid 3D shape hypothesis generation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from ..core.contracts import ReconstructionHypothesis
from ..core.registry import MODEL_REGISTRY, ModelSpec
from ..core.types import ArtifactType, ModelCapability, PrecisionType
from ..representations.mesh import MeshData
from .base_worker import Base3DWorker


TRIPOSR_SPEC = ModelSpec(
    name="TripoSR",
    version="1.0.0",
    capability=ModelCapability.RAPID_HYPOTHESIS_GENERATION,
    input_types=[ArtifactType.INPUT_MEDIA, ArtifactType.OBSERVATION],
    output_types=[ArtifactType.MESH, ArtifactType.RECONSTRUCTION_HYPOTHESIS],
    estimated_vram_bytes=int(6.0 * (1024 ** 3)),
    supported_precisions=[PrecisionType.FP16, PrecisionType.FP32],
    description="Fast feedforward reconstruction worker for initial geometry hypothesis.",
)


@MODEL_REGISTRY.register("TripoSR", spec=TRIPOSR_SPEC)
class TripoSRWorker(Base3DWorker):
    """TripoSR image-to-3D hypothesis worker."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(TRIPOSR_SPEC, config)

    def initialize(self, device: str = "cuda", precision: str = "fp16") -> None:
        self.device = device
        self.precision = precision
        # Actual weights loaded in Phase 2
        self.runtime_state = "loaded_resident"

    def infer(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.generate_hypothesis(inputs[0] if inputs else None)

    def generate_hypothesis(self, input_artifact: Any) -> Tuple[MeshData, ReconstructionHypothesis]:
        hypothesis = ReconstructionHypothesis(
            worker_name=self.spec.name,
            completeness_score=0.75,
            surface_smoothness_score=0.80,
            geometric_evidence_agreement_score=0.70,
            ranking_score=0.75,
        )
        # Empty placeholder mesh for contract validation
        mesh = MeshData(
            vertices=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int32),
        )
        return mesh, hypothesis
