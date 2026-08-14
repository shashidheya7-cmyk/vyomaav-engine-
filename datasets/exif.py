"""
VYOMAAV Base Model Engine
Module: dataset.exif

Parses raw camera EXIF metadata (focal length, sensor dimensions, resolution, aperture)
and calculates metric focal lengths and the 3x3 Camera Intrinsics Matrix K.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class CameraMetadata:
    """Raw physical camera parameters extracted from image/video EXIF headers."""
    focal_length_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    image_width_px: int
    image_height_px: int
    iso: int = 100
    exposure_time_s: float = 0.01
    rolling_shutter_readout_s: float = 0.0
    timestamp_s: float = 0.0

    def compute_intrinsics_k(self) -> List[float]:
        """Calculates 3x3 Intrinsic Matrix K (returned as a flattened 9-element list).

        f_x = (f_mm * W_px) / W_sensor_mm
        f_y = (f_mm * H_px) / H_sensor_mm
        c_x = W_px / 2.0
        c_y = H_px / 2.0

        K = [f_x,  0 , c_x,
              0 , f_y, c_y,
              0 ,  0 ,  1  ]
        """
        f_x = (self.focal_length_mm * self.image_width_px) / self.sensor_width_mm
        f_y = (self.focal_length_mm * self.image_height_px) / self.sensor_height_mm
        c_x = self.image_width_px / 2.0
        c_y = self.image_height_px / 2.0

        return [
            f_x, 0.0, c_x,
            0.0, f_y, c_y,
            0.0, 0.0, 1.0
        ]

    def compute_field_of_view_degrees(self) -> Tuple[float, float]:
        """Calculates horizontal and vertical Field of View (FOV) in degrees."""
        import math
        f_x, _, c_x, _, f_y, c_y, _, _, _ = self.compute_intrinsics_k()
        fov_h = 2.0 * math.atan(self.image_width_px / (2.0 * f_x)) * (180.0 / math.pi)
        fov_v = 2.0 * math.atan(self.image_height_px / (2.0 * f_y)) * (180.0 / math.pi)
        return fov_h, fov_v


class EXIFExtractor:
    """Standardized metadata extraction interface."""

    @staticmethod
    def create_simulated_metadata(
        resolution: Tuple[int, int] = (1920, 1080),
        focal_length_mm: float = 24.0,
        sensor_type: str = "full_frame"
    ) -> CameraMetadata:
        """Helper to create calibrated metadata for synthetic or uncalibrated sources."""
        sensor_dims = {
            "full_frame": (36.0, 24.0),
            "aps_c": (23.5, 15.6),
            "iphone_main": (7.0, 5.25),
            "drone_1inch": (13.2, 8.8)
        }
        s_w, s_h = sensor_dims.get(sensor_type, (36.0, 24.0))
        return CameraMetadata(
            focal_length_mm=focal_length_mm,
            sensor_width_mm=s_w,
            sensor_height_mm=s_h,
            image_width_px=resolution[0],
            image_height_px=resolution[1]
        )