import numpy as np
from typing import Dict, Any, List
from vyomaa.camera_geometry.base import BaseGeometryBackend
from vyomaa.multiview.contracts import ViewSet, GeometryEvidence, CameraEstimate, DenseGeometry, CorrespondenceSet

class AnalyticFallbackAdapter(BaseGeometryBackend):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.is_initialized = True

    def initialize(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def estimate_geometry(self, view_set: ViewSet) -> GeometryEvidence:
        num_views = len(view_set.observation_ids)
        cameras: List[CameraEstimate] = []
        dense_geometries: List[DenseGeometry] = []
        correspondences: List[CorrespondenceSet] = []

        for idx, obs_id in enumerate(view_set.observation_ids):
            k = np.array([[500.0, 0.0, 256.0], [0.0, 500.0, 256.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            rt = np.eye(4, dtype=np.float32)[:3, :]
            rt[0, 3] = float(idx * 0.1)

            cam = CameraEstimate(
                camera_id=obs_id,
                intrinsics_k=k,
                extrinsics_rt=rt,
                focal_lengths=(500.0, 500.0),
                principal_point=(256.0, 256.0),
                backend_name="analytic_fallback",
                coordinate_convention="opencv",
                confidence=0.50,
                validity_state=True,
                provenance={"backend": "analytic_fallback", "warning": "FALLBACK_GEOMETRY_NOT_MODEL_INFERENCE"}
            )
            cameras.append(cam)
            dense_geometries.append(DenseGeometry(
                source_observation_id=obs_id,
                depth_array_shape=(512, 512),
                point_map_shape=(512, 512, 3),
                validity_mask_shape=(512, 512),
                confidence=0.50,
                resolution=(512, 512),
                backend="analytic_fallback"
            ))

        return GeometryEvidence(
            backend="analytic_fallback",
            cameras=cameras,
            dense_geometry=dense_geometries,
            correspondences=correspondences,
            confidence=0.50,
            reprojection_metrics={"mean_error": 0.0},
            consistency_metrics={"multiview_consistency": 0.50},
            warnings=["PROCESSED_WITH_ANALYTIC_FALLBACK_NOT_VGGT"],
            provenance={"backend": "analytic_fallback", "real_inference": False}
        )

    def release(self) -> None:
        pass

    def capabilities(self) -> Dict[str, Any]:
        return {"backend": "analytic_fallback", "real_inference": False}
