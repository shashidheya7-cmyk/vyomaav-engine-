"""Texture artifact representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..core.base_artifact import BaseArtifact
from ..core.metadata import ArtifactMetadata
from ..core.provenance import ProvenanceRecord
from ..core.types import ArtifactType


@dataclass
class TextureArtifact(BaseArtifact):
    """2D texture map representation with color space and channel semantics."""

    texture_role: str = "albedo"  # albedo, roughness, metallic, normal, occlusion, emissive
    width: int = 2048
    height: int = 2048
    channels: int = 3
    color_space: str = "sRGB"  # sRGB or Linear
    storage_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.artifact_type = ArtifactType.TEXTURE
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "texture_role": self.texture_role,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "color_space": self.color_space,
            "storage_path": self.storage_path,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TextureArtifact:
        kwargs = dict(data)
        kwargs["artifact_type"] = ArtifactType.TEXTURE
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = ArtifactMetadata.from_dict(kwargs["metadata"])
        if "provenance" in kwargs and isinstance(kwargs["provenance"], dict):
            kwargs["provenance"] = ProvenanceRecord.from_dict(kwargs["provenance"])
        return cls(**kwargs)
