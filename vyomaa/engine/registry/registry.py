
"""Thread-safe named implementation registry."""

from threading import RLock
from typing import Callable, TypeVar

from ..core.exceptions import RegistryError

T = TypeVar("T")


class Registry:
    """Registry supporting explicit, collision-safe decorator registration."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._modules: dict[str, object] = {}
        self._lock = RLock()

    def register_module(self, name: str) -> Callable[[T], T]:
        """Register an implementation under *name* and return it unchanged."""
        if not name or not name.strip():
            raise RegistryError(f"{self.name} registry requires a non-empty name")

        def decorator(module: T) -> T:
            with self._lock:
                if name in self._modules:
                    raise RegistryError(f"{self.name} provider already registered: {name}")
                self._modules[name] = module
            return module

        return decorator

    def get(self, name: str) -> object:
        """Return a registered implementation or raise a domain-specific error."""
        with self._lock:
            try:
                return self._modules[name]
            except KeyError as exc:
                options = ", ".join(sorted(self._modules)) or "none"
                raise RegistryError(f"unknown {self.name} provider '{name}'; available: {options}") from exc


FUSION_REGISTRY = Registry("FUSION")


