"""Provenance tracking models for lineage, auditing, and multi-source attribution."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class ProvenanceRecord:
    """Immutable audit trail record for an artifact's creation and transformation."""

    producer_subsystem: str
    producer_version: str = "1.0.0"
    producer_model: Optional[str] = None
    parent_artifact_ids: List[str] = field(default_factory=list)
    generation_parameters: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    operator_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert provenance record to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProvenanceRecord:
        """Construct provenance record from dictionary."""
        return cls(**data)

    def add_parent(self, parent_id: str) -> None:
        """Register a parent artifact dependency."""
        if parent_id not in self.parent_artifact_ids:
            self.parent_artifact_ids.append(parent_id)
