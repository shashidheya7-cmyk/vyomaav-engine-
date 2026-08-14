
"""PyTorch import boundary with a NumPy fallback for CPU-only verification."""

from __future__ import annotations

import numpy as np

try:  # Prefer the real differentiable runtime whenever installed.
    import torch as torch
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    class _Tensor:
        """Small CPU tensor adapter used solely to keep validation dependency-light."""

        def __init__(self, values: object) -> None:
            self._array = np.asarray(values, dtype=np.float32)

        @property
        def shape(self) -> tuple[int, ...]: return self._array.shape
        @property
        def ndim(self) -> int: return self._array.ndim
        def detach(self) -> "_Tensor": return self
        def to(self, **_: object) -> "_Tensor": return self
        def numpy(self) -> np.ndarray: return self._array
        def __array__(self, dtype: object = None) -> np.ndarray: return np.asarray(self._array, dtype=dtype)

    class _Cuda:
        @staticmethod
        def is_available() -> bool: return False
        @staticmethod
        def empty_cache() -> None: return None

    class _TorchFallback:
        Tensor = _Tensor
        float32 = np.float32
        cuda = _Cuda()
        _rng = np.random.default_rng()

        @classmethod
        def manual_seed(cls, seed: int) -> None: cls._rng = np.random.default_rng(seed)
        @classmethod
        def rand(cls, *shape: int) -> _Tensor: return _Tensor(cls._rng.random(shape, dtype=np.float32))
        @staticmethod
        def stack(values: list[_Tensor], dim: int = 0) -> _Tensor: return _Tensor(np.stack([v.numpy() for v in values], axis=dim))
        @staticmethod
        def from_numpy(value: np.ndarray) -> _Tensor: return _Tensor(value)

    torch = _TorchFallback()


