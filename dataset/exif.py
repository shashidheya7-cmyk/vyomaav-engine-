"""EXIF Metadata Extractor."""
from dataclasses import dataclass
from typing import Tuple, Optional, List
import math

@dataclass
class CameraMetadata:
    focal_length_mm: float = 24.0
    sensor_width_mm: float = 36.0
    sensor_height_mm: float = 24.0
    image_width_px: int = 1920
    image_height_px: int = 1080
    resolution: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if self.resolution is None:
            self.resolution = (self.image_width_px, self.image_height_px)

    def compute_intrinsics_k(self) -> List[float]:
        fx = (self.focal_length_mm * self.image_width_px) / self.sensor_width_mm
        fy = (self.focal_length_mm * self.image_height_px) / self.sensor_height_mm
        cx = self.image_width_px / 2.0
        cy = self.image_height_px / 2.0
        return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

    def compute_field_of_view_degrees(self) -> Tuple[float, float]:
        fov_h = 2 * math.atan(self.sensor_width_mm / (2 * self.focal_length_mm)) * (180.0 / math.pi)
        fov_v = 2 * math.atan(self.sensor_height_mm / (2 * self.focal_length_mm)) * (180.0 / math.pi)
        return fov_h, fov_v

class EXIFExtractor:
    @staticmethod
    def create_simulated_metadata(resolution=(1920, 1080), focal_length_mm=24.0) -> CameraMetadata:
        return CameraMetadata(
            focal_length_mm=focal_length_mm,
            image_width_px=resolution[0],
            image_height_px=resolution[1],
            resolution=resolution
        )
