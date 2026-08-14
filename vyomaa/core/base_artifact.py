"""Base artifact abstraction defining canonical identity, typing, serialization, and validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from typing import Any, Dict, Optional, Type, TypeVar
import uuid

from .exceptions import SchemaValidationError
from .metadata import ArtifactMetadata
from .provenance import ProvenanceRecord
from .types import ArtifactType, ConfidenceLevel

T = TypeVar("T", bound="BaseArtifact")


@dataclass
class BaseArtifact(ABC):
    """Abstract base artifact guaranteeing id, type, provenance, confidence, and metadata."""

    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: ArtifactType = field(default=ArtifactType.INPUT_MEDIA)
    name: str = "unnamed_artifact"
    confidence_score: float = 1.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)
    provenance: ProvenanceRecord = field(
        default_factory=lambda: ProvenanceRecord(producer_subsystem="core_framework")
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate artifact invariants and confidence score bounds."""
        if not self.artifact_id or not isinstance(self.artifact_id, str):
            raise SchemaValidationError(f"Invalid artifact_id: {self.artifact_id}")
        if not isinstance(self.artifact_type, ArtifactType):
            if isinstance(self.artifact_type, str):
                self.artifact_type = ArtifactType(self.artifact_type)
            else:
                raise SchemaValidationError(f"Invalid artifact_type: {self.artifact_type}")
        if not (0.0 <= self.confidence_score <= 1.0):
            raise SchemaValidationError(
                f"Confidence score must be in [0.0, 1.0], got {self.confidence_score}"
            )
        self._sync_confidence_level()

    def _sync_confidence_level(self) -> None:
        """Compute categorical confidence level from numerical score."""
        if self.confidence_score >= 0.99:
            self.confidence_level = ConfidenceLevel.GROUND_TRUTH
        elif self.confidence_score >= 0.8:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence_score >= 0.5:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.confidence_score >= 0.2:
            self.confidence_level = ConfidenceLevel.LOW
        else:
            self.confidence_level = ConfidenceLevel.VERY_LOW

    def to_dict(self) -> Dict[str, Any]:
        """Serialize base artifact properties to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "name": self.name,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
            "metadata": self.metadata.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize artifact to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Reconstruct artifact instance from dictionary."""
        raise NotImplementedError
