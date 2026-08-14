"""Artifact Manifest schema for indexing, provenance auditing, and storage integrity."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional

from ..core.types import ArtifactType


@dataclass
class ArtifactManifest:
    """Canonical descriptor indexing stored artifact files and dependencies."""

    artifact_id: str
    artifact_type: str
    file_path: str
    producer_subsystem: str
    producer_version: str = "1.0.0"
    producer_model: Optional[str] = None
    parent_artifact_ids: List[str] = field(default_factory=list)
    checksum_sha256: str = ""
    schema_version: str = "1.0.0"
    confidence_score: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ArtifactManifest:
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> ArtifactManifest:
        return cls.from_dict(json.loads(json_str))
