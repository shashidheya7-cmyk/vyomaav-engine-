
"""PBR texture state."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TextureData:
    """Optional PBR texture maps represented as NumPy arrays."""

    albedo: Optional[np.ndarray] = None
    roughness: Optional[np.ndarray] = None
    metallic: Optional[np.ndarray] = None
    normal: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        for name in ("albedo", "roughness", "metallic", "normal"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.ascontiguousarray(value, dtype=np.float32))


