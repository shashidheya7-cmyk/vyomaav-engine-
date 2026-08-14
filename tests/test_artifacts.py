"""Unit tests for artifact serialization and schema validation."""

import json
import unittest
import numpy as np

from vyomaa.core.contracts import (
    InputArtifact,
    Camera,
    Observation,
    DepthMap,
    SegmentationMask,
    ReconstructionHypothesis,
    ValidationReport,
)
from vyomaa.core.exceptions import SchemaValidationError
from vyomaa.core.types import ArtifactType, ModalityType, ConfidenceLevel


class TestArtifactSerialization(unittest.TestCase):

    def test_input_artifact_serialization(self):
        art = InputArtifact(
            name="Source Photo",
            modality=ModalityType.RGB_IMAGE,
            file_path="/tmp/image.png",
            resolution=(1920, 1080),
            confidence_score=0.95,
        )
        d = art.to_dict()
        self.assertEqual(d["artifact_type"], ArtifactType.INPUT_MEDIA.value)
        self.assertEqual(d["modality"], ModalityType.RGB_IMAGE.value)
        self.assertEqual(d["confidence_level"], ConfidenceLevel.HIGH.value)

        # Reconstruct
        art_re = InputArtifact.from_dict(d)
        self.assertEqual(art_re.artifact_id, art.artifact_id)
        self.assertEqual(art_re.resolution, (1920, 1080))
        self.assertEqual(art_re.confidence_score, 0.95)

    def test_schema_validation_confidence_bounds(self):
        with self.assertRaises(SchemaValidationError):
            InputArtifact(confidence_score=1.5)

        with self.assertRaises(SchemaValidationError):
            InputArtifact(confidence_score=-0.1)

    def test_validation_report_serialization(self):
        report = ValidationReport(
            name="Stage 15 Quality Audit",
            is_valid=True,
            overall_quality_score=0.92,
            is_watertight=True,
            is_manifold=True,
            num_holes_detected=0,
            warnings=["Slight lighting asymmetry"],
        )
        d = report.to_dict()
        reconstructed = ValidationReport.from_dict(d)
        self.assertTrue(reconstructed.is_valid)
        self.assertEqual(reconstructed.overall_quality_score, 0.92)
        self.assertEqual(len(reconstructed.warnings), 1)


if __name__ == "__main__":
    unittest.main()
