"""
VYOMAAV Base Model Engine
Test Suite: tests/test_dataset.py

Pytest suite validating EXIF camera metadata processing, intrinsic matrix K computation,
FOV calculation, and 2D-to-3D bounding box spatial projection.
"""

import pytest
from dataset.exif import CameraMetadata, EXIFExtractor
from dataset.ingest import MultimodalIngestionPipeline, RawSensorSample, BoundingBox2D


def test_exif_intrinsics_and_fov_calculation():
    # 36mm sensor, 24mm lens, 1920x1080 resolution
    # f_x = (24 * 1920) / 36 = 1280.0
    # c_x = 1920 / 2 = 960.0, c_y = 1080 / 2 = 540.0
    meta = CameraMetadata(
        focal_length_mm=24.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        image_width_px=1920,
        image_height_px=1080
    )

    k = meta.compute_intrinsics_k()
    assert len(k) == 9
    assert k[0] == 1280.0  # f_x
    assert k[4] == 1080.0  # f_y
    assert k[2] == 960.0   # c_x
    assert k[5] == 540.0   # c_y

    fov_h, fov_v = meta.compute_field_of_view_degrees()
    assert 73.0 < fov_h < 74.0


def test_2d_to_3d_bbox_projection():
    pipeline = MultimodalIngestionPipeline(default_depth_m=4.0)
    meta = EXIFExtractor.create_simulated_metadata(resolution=(1920, 1080), focal_length_mm=24.0)

    # 2D detection centered in frame
    box2d = BoundingBox2D(
        label="chair",
        class_id=1,
        confidence=0.92,
        x_min_px=860.0,
        y_min_px=440.0,
        x_max_px=1060.0,
        y_max_px=640.0
    )

    b_min, b_max = pipeline.project_2d_to_3d_bbox(box2d, meta, metric_depth_m=4.0)

    assert len(b_min) == 3 and len(b_max) == 3
    assert b_min[2] == 3.5  # depth - 0.5
    assert b_max[2] == 4.5  # depth + 0.5
    assert b_min[0] < b_max[0]
    assert b_min[1] < b_max[1]


def test_sample_ingestion_pipeline():
    pipeline = MultimodalIngestionPipeline(default_depth_m=3.0)
    meta = EXIFExtractor.create_simulated_metadata()

    sample = RawSensorSample(
        sample_id="frame_001",
        timestamp_s=100.5,
        camera_metadata=meta,
        detections_2d=[
            BoundingBox2D("table", 101, 0.95, 100.0, 100.0, 500.0, 500.0),
            BoundingBox2D("lamp", 102, 0.88, 600.0, 200.0, 700.0, 400.0)
        ]
    )

    observations = pipeline.process_sample_to_observations(sample)

    assert len(observations) == 2
    assert observations[0].label == "table"
    assert observations[0].confidence == 0.95
    assert observations[1].label == "lamp"