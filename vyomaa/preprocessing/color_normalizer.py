"""Color space transformation and radiometric normalization without source overwrite."""

from __future__ import annotations

from typing import Tuple
import numpy as np
from PIL import Image

from ..core.exceptions import PreprocessingError
from ..core.provenance import ProvenanceRecord


class ColorNormalizer:
    """Provides non-destructive color space transformations (sRGB, Linear RGB, ACEScg)."""

    @staticmethod
    def srgb_to_linear(image_array: np.ndarray) -> np.ndarray:
        """Convert uint8 or float32 sRGB array in [0, 1] to linear radiometric RGB."""
        arr = image_array.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0

        # Exact IEC 61966-2-1 transfer function
        linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
        return np.clip(linear, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def linear_to_srgb(linear_array: np.ndarray) -> np.ndarray:
        """Convert float32 linear RGB array to gamma-corrected sRGB uint8."""
        arr = np.clip(linear_array, 0.0, 1.0)
        srgb = np.where(arr <= 0.0031308, arr * 12.92, 1.055 * (arr ** (1.0 / 2.4)) - 0.055)
        return (np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    @staticmethod
    def srgb_to_aces(image_array: np.ndarray) -> np.ndarray:
        """Transform sRGB to ACEScg color space via standard color matrix."""
        linear = ColorNormalizer.srgb_to_linear(image_array)
        # sRGB (D65) to ACEScg matrix
        M_srgb_to_aces = np.array([
            [0.613097, 0.339523, 0.047379],
            [0.070194, 0.916354, 0.013452],
            [0.020616, 0.109570, 0.869814]
        ], dtype=np.float32)
        aces = np.einsum("ij,...j->...i", M_srgb_to_aces, linear)
        return np.clip(aces, 0.0, 1.0).astype(np.float32)
