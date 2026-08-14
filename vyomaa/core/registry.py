"""Generic model and module registry defining adapter contracts and capability schemas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from .exceptions import RegistryError
from .types import ArtifactType, DeviceType, ModelCapability, ModelRuntimeState, PrecisionType

T = TypeVar("T")


@dataclass
class ModelSpec:
    """Metadata specification describing a worker model's capabilities and resource requirements."""

    name: str
    version: str = "1.0.0"
    capability: ModelCapability = ModelCapability.HIGH_DETAIL_3D_GENERATION
    input_types: List[ArtifactType] = field(default_factory=lambda: [ArtifactType.INPUT_MEDIA])
    output_types: List[ArtifactType] = field(default_factory=lambda: [ArtifactType.MESH])
    estimated_vram_bytes: int = int(8.0 * (1024 ** 3))  # 8 GB default
    supported_precisions: List[PrecisionType] = field(default_factory=lambda: [PrecisionType.FP16, PrecisionType.FP32])
    supports_batching: bool = False
    device_requirements: List[DeviceType] = field(default_factory=lambda: [DeviceType.CUDA, DeviceType.CPU])
    description: str = ""

    @property
    def estimated_vram_gb(self) -> float:
        return round(self.estimated_vram_bytes / (1024 ** 3), 2)


class ModelAdapter(ABC):
    """Abstract worker adapter defining the contract between VYOMAA and specialized neural models."""

    def __init__(self, spec: ModelSpec, config: Optional[Dict[str, Any]] = None) -> None:
        self.spec = spec
        self.config = config or {}
        self.runtime_state = ModelRuntimeState.UNINITIALIZED

    @abstractmethod
    def initialize(self, device: str = "cuda", precision: str = "fp16") -> None:
        """Allocate model weights and move to target compute device."""
        raise NotImplementedError

    @abstractmethod
    def infer(self, *inputs: Any, **kwargs: Any) -> Any:
        """Execute deterministic forward pass."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release allocated tensors or state when explicitly requested."""
        self.runtime_state = ModelRuntimeState.EVICTED


class Registry:
    """Thread-safe generic registry for models, pipeline tasks, and subsystem providers."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Any] = {}
        self._specs: Dict[str, ModelSpec] = {}
        self._lock = RLock()

    def register(self, name: str, spec: Optional[ModelSpec] = None) -> Callable[[T], T]:
        """Decorator to register an implementation under *name*."""
        if not name or not name.strip():
            raise RegistryError(f"{self.name} registry requires a non-empty name")

        def decorator(cls_or_fn: T) -> T:
            with self._lock:
                if name in self._items:
                    raise RegistryError(f"{self.name} already contains a registered item for '{name}'")
                self._items[name] = cls_or_fn
                if spec is not None:
                    self._specs[name] = spec
            return cls_or_fn

        return decorator

    def get(self, name: str) -> Any:
        """Retrieve registered item by name."""
        with self._lock:
            if name not in self._items:
                available = ", ".join(sorted(self._items.keys())) or "none"
                raise RegistryError(f"'{name}' not registered in {self.name}; available: [{available}]")
            return self._items[name]

    def get_spec(self, name: str) -> Optional[ModelSpec]:
        """Retrieve model spec if registered."""
        with self._lock:
            return self._specs.get(name)

    def list_all(self) -> List[str]:
        """List all registered identifiers."""
        with self._lock:
            return sorted(self._items.keys())

    def find_by_capability(self, capability: ModelCapability) -> List[str]:
        """Find models supporting a specific capability."""
        with self._lock:
            return [name for name, spec in self._specs.items() if spec.capability == capability]


# Global Registries
MODEL_REGISTRY = Registry("MODEL_REGISTRY")
TASK_REGISTRY = Registry("TASK_REGISTRY")
