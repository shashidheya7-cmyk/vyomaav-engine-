"""Metadata structures for artifact schema versioning, checksums, and tags."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional


@dataclass
class ArtifactMetadata:
    """Comprehensive metadata container for engine artifacts."""

    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    checksum_sha256: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    storage_uri: Optional[str] = None
    mime_type: Optional[str] = None
    byte_size: Optional[int] = None
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ArtifactMetadata:
        """Create metadata from dictionary."""
        return cls(**data)

    def update_timestamp(self) -> None:
        """Refresh the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def compute_sha256(data_bytes: bytes) -> str:
        """Calculate standard SHA-256 digest of payload."""
        return hashlib.sha256(data_bytes).hexdigest()
