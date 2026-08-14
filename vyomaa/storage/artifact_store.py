"""Artifact Store managing structured workspace layouts, manifests, and version resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar
import uuid

from ..core.base_artifact import BaseArtifact
from ..core.exceptions import ArtifactStoreError
from ..core.metadata import ArtifactMetadata
from ..core.types import ArtifactType
from .manifest import ArtifactManifest

T = TypeVar("T", bound=BaseArtifact)


class ArtifactStore:
    """Filesystem-backed artifact storage engine maintaining typed directories and manifests."""

    SUBDIRECTORIES = {
        ArtifactType.INPUT_MEDIA: "observations",
        ArtifactType.MEDIA_SEQUENCE: "observations",
        ArtifactType.FRAME: "observations",
        ArtifactType.OBSERVATION: "observations",
        ArtifactType.DEPTH_MAP: "depth",
        ArtifactType.SEGMENTATION_MASK: "observations",
        ArtifactType.CAMERA: "cameras",
        ArtifactType.CAMERA_TRAJECTORY: "cameras",
        ArtifactType.POINT_CLOUD: "pointclouds",
        ArtifactType.MESH: "meshes",
        ArtifactType.GAUSSIAN_SPLAT: "pointclouds",
        ArtifactType.SDF_VOLUME: "meshes",
        ArtifactType.PBR_MATERIAL: "materials",
        ArtifactType.TEXTURE: "textures",
        ArtifactType.OBJECT_ENTITY: "scenes",
        ArtifactType.SCENE_GRAPH: "scenes",
        ArtifactType.WORLD_GRAPH: "worlds",
        ArtifactType.RECONSTRUCTION_HYPOTHESIS: "meshes",
        ArtifactType.CONFIDENCE_MAP: "validation",
        ArtifactType.VALIDATION_REPORT: "validation",
    }

    def __init__(self, workspace_root: str = "workspace") -> None:
        self.root = Path(workspace_root).resolve()
        self.artifacts_dir = self.root / "artifacts"
        self.manifests_dir = self.root / "manifests"
        self._initialize_layout()

    def _initialize_layout(self) -> None:
        """Create all canonical directory structures."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        for sub in set(self.SUBDIRECTORIES.values()):
            (self.artifacts_dir / sub).mkdir(parents=True, exist_ok=True)

    def save_artifact(self, artifact: BaseArtifact) -> ArtifactManifest:
        """Serialize artifact to JSON, compute SHA-256 digest, and store its manifest."""
        subfolder = self.SUBDIRECTORIES.get(artifact.artifact_type, "observations")
        target_dir = self.artifacts_dir / subfolder
        artifact_filename = f"{artifact.artifact_id}.json"
        artifact_path = target_dir / artifact_filename

        artifact.metadata.storage_uri = str(artifact_path)
        data_dict = artifact.to_dict()
        data_json = json.dumps(data_dict, indent=2, default=str)
        data_bytes = data_json.encode("utf-8")

        # Compute checksum of final serialized payload
        checksum = ArtifactMetadata.compute_sha256(data_bytes)
        artifact.metadata.checksum_sha256 = checksum
        artifact.metadata.byte_size = len(data_bytes)

        # Write artifact payload
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write(data_json)

        # Create and write manifest
        manifest = ArtifactManifest(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type.value,
            file_path=str(artifact_path),
            producer_subsystem=artifact.provenance.producer_subsystem,
            producer_version=artifact.provenance.producer_version,
            producer_model=artifact.provenance.producer_model,
            parent_artifact_ids=list(artifact.provenance.parent_artifact_ids),
            checksum_sha256=checksum,
            schema_version=artifact.metadata.schema_version,
            confidence_score=artifact.confidence_score,
            created_at=artifact.metadata.created_at,
            metadata=artifact.metadata.to_dict(),
        )

        manifest_path = self.manifests_dir / f"{artifact.artifact_id}.manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.to_json(indent=2))

        return manifest

    def load_manifest(self, artifact_id: str) -> ArtifactManifest:
        """Load manifest metadata by artifact ID."""
        manifest_path = self.manifests_dir / f"{artifact_id}.manifest.json"
        if not manifest_path.is_file():
            raise ArtifactStoreError(f"Manifest for artifact '{artifact_id}' not found at {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            return ArtifactManifest.from_json(f.read())

    def load_artifact_data(self, artifact_id: str) -> Dict[str, Any]:
        """Load and verify raw JSON payload for an artifact."""
        manifest = self.load_manifest(artifact_id)
        path = Path(manifest.file_path)
        if not path.is_file():
            raise ArtifactStoreError(f"Artifact file missing at {path}")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check integrity
        actual_hash = ArtifactMetadata.compute_sha256(content.encode("utf-8"))
        if actual_hash != manifest.checksum_sha256:
            raise ArtifactStoreError(
                f"Checksum mismatch for artifact '{artifact_id}': expected {manifest.checksum_sha256}, got {actual_hash}"
            )
        return json.loads(content)

    def load_artifact(self, artifact_id: str, artifact_cls: Type[T]) -> T:
        """Load, verify, and deserialize artifact into its target typed class."""
        data = self.load_artifact_data(artifact_id)
        return artifact_cls.from_dict(data)

    def list_manifests(self, artifact_type: Optional[ArtifactType] = None) -> List[ArtifactManifest]:
        """Query manifests with optional type filter."""
        results = []
        for m_file in self.manifests_dir.glob("*.manifest.json"):
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    man = ArtifactManifest.from_json(f.read())
                    if artifact_type is None or man.artifact_type == artifact_type.value:
                        results.append(man)
            except Exception:
                continue
        return results

    def get_lineage(self, artifact_id: str) -> List[str]:
        """Retrieve recursive parent artifact ID dependency chain."""
        lineage: List[str] = []
        visited = set()

        def _trace(curr_id: str) -> None:
            if curr_id in visited:
                return
            visited.add(curr_id)
            try:
                man = self.load_manifest(curr_id)
                for pid in man.parent_artifact_ids:
                    lineage.append(pid)
                    _trace(pid)
            except Exception:
                pass

        _trace(artifact_id)
        return lineage
