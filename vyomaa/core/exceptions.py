"""Canonical typed exception hierarchy for the VYOMAA Engine."""

from __future__ import annotations


class VyomaaError(Exception):
    """Base exception class for all VYOMAA engine errors."""
    pass


class ConfigurationError(VyomaaError):
    """Raised when engine or subsystem configuration is invalid or missing."""
    pass


class SchemaValidationError(VyomaaError):
    """Raised when an artifact or contract fails schema validation."""
    pass


class ArtifactStoreError(VyomaaError):
    """Raised when storing, loading, or indexing an artifact fails."""
    pass


class SceneGraphError(VyomaaError):
    """Raised when scene or world graph operations violate invariants."""
    pass


class GeometryError(VyomaaError):
    """Raised when geometric or topological operations fail."""
    pass


class HardwareError(VyomaaError):
    """Raised when GPU or VRAM operations fail or exceed limits."""
    pass


class RegistryError(VyomaaError):
    """Raised when model or module registration/lookup fails."""
    pass


class PipelineExecutionError(VyomaaError):
    """Raised when DAG pipeline execution encounters an unrecoverable failure."""
    pass


class TaskDependencyError(PipelineExecutionError):
    """Raised when task graph dependencies are circular or unsatisfied."""
    pass
