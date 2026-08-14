"""Aspect ratio padder and coordinate transformation tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from PIL import Image

from ..core.exceptions import PreprocessingError


@dataclass
class AspectRatioMetadata:
    """Tracks coordinate shifts and scale factors introduced during aspect ratio padding."""

    original_width: int
    original_height: int
    target_width: int
    target_height: int
    scale_factor: float
    pad_left: int
    pad_top: int


class AspectRatioNormalizer:
    """Pads or scales images to target dimensions while maintaining aspect ratio and tracking coordinates."""

    @staticmethod
    def pad_to_target(
        image: Image.Image,
        target_size: Tuple[int, int] = (1024, 1024),
        fill_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> Tuple[Image.Image, AspectRatioMetadata]:
        """Pad image to target_size (width, height) with border padding and return metadata."""
        orig_w, orig_h = image.size
        tgt_w, tgt_h = target_size

        scale = min(tgt_w / orig_w, tgt_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        pad_left = (tgt_w - new_w) // 2
        pad_top = (tgt_h - new_h) // 2

        padded = Image.new("RGB", (tgt_w, tgt_h), fill_color)
        padded.paste(resized, (pad_left, pad_top))

        meta = AspectRatioMetadata(
            original_width=orig_w,
            original_height=orig_h,
            target_width=tgt_w,
            target_height=tgt_h,
            scale_factor=scale,
            pad_left=pad_left,
            pad_top=pad_top,
        )
        return padded, meta
