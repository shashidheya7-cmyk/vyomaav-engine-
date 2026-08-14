"""Metadata and EXIF extractor for image and video media."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image, ExifTags

from ..core.exceptions import IngestionError


class MetadataExtractor:
    """Extracts camera, lens, optical, and timestamp metadata from media files."""

    @staticmethod
    def extract_image_exif(image_path: str) -> Dict[str, Any]:
        """Extract camera make, model, focal length, exposure, and orientation from EXIF tags."""
        p = Path(image_path)
        if not p.is_file():
            raise IngestionError(f"Image file does not exist: {image_path}")

        meta: Dict[str, Any] = {
            "file_name": p.name,
            "file_size_bytes": p.stat().st_size,
            "format": None,
            "camera_make": None,
            "camera_model": None,
            "focal_length_mm": None,
            "focal_length_35mm": None,
            "iso": None,
            "exposure_time": None,
            "f_number": None,
            "orientation": 1,
            "datetime_original": None,
        }

        try:
            with Image.open(p) as img:
                meta["format"] = img.format
                meta["width"], meta["height"] = img.size
                meta["mode"] = img.mode

                exif_data = img._getexif() if hasattr(img, "_getexif") and img._getexif() else None
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        if tag == "Make":
                            meta["camera_make"] = str(value).strip()
                        elif tag == "Model":
                            meta["camera_model"] = str(value).strip()
                        elif tag == "FocalLength":
                            meta["focal_length_mm"] = float(value) if isinstance(value, (int, float)) else float(value[0])/float(value[1]) if isinstance(value, tuple) else None
                        elif tag == "FocalLengthIn35mmFilm":
                            meta["focal_length_35mm"] = float(value)
                        elif tag == "ISOSpeedRatings":
                            meta["iso"] = int(value) if isinstance(value, (int, float)) else int(value[0]) if isinstance(value, tuple) else None
                        elif tag == "ExposureTime":
                            meta["exposure_time"] = float(value) if isinstance(value, (int, float)) else float(value[0])/float(value[1]) if isinstance(value, tuple) else None
                        elif tag == "FNumber":
                            meta["f_number"] = float(value) if isinstance(value, (int, float)) else float(value[0])/float(value[1]) if isinstance(value, tuple) else None
                        elif tag == "Orientation":
                            meta["orientation"] = int(value)
                        elif tag == "DateTimeOriginal":
                            meta["datetime_original"] = str(value)
        except Exception as exc:
            meta["exif_warning"] = str(exc)

        return meta
