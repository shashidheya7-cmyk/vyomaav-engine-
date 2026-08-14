import unittest
import numpy as np
from vyomaa.multiview.contracts import ViewSet, CameraEstimate, FusedWorldGeometry
from vyomaa.multiview.keyframe_selector import KeyframeSelector
from vyomaa.multiview.view_graph import ViewGraph
from vyomaa.validation.camera_validation import CameraValidator
from vyomaa.fusion.dense_point_fusion import DensePointFusion
from vyomaa.fusion.outlier_rejection import OutlierRejection

class TestMultiViewFoundation(unittest.TestCase):
    def test_view_set_serialization(self):
        vs = ViewSet(
            observation_ids=["obs_1", "obs_2"],
            timestamps=[0.0, 1.0],
            keyframe_flags=[True, True],
            image_paths=["/tmp/1.jpg", "/tmp/2.jpg"]
        )
        d = vs.to_dict()
        self.assertEqual(d["observation_ids"], ["obs_1", "obs_2"])
        self.assertTrue(d["keyframe_flags"][0])

    def test_keyframe_selector(self):
        selector = KeyframeSelector(temporal_window=2)
        obs_ids = [f"obs_{i}" for i in range(10)]
        ts = [float(i) * 0.1 for i in range(10)]
        paths = [f"/tmp/{i}.jpg" for i in range(10)]
        
        vs = selector.select_keyframes(obs_ids, ts, paths)
        self.assertGreater(len(vs.observation_ids), 0)
        self.assertLessEqual(len(vs.observation_ids), 10)

    def test_view_graph_construction(self):
        vs = ViewSet(
            observation_ids=["obs_1", "obs_2", "obs_3"],
            timestamps=[0.0, 1.0, 2.0],
            keyframe_flags=[True, True, True],
            image_paths=["a.jpg", "b.jpg", "c.jpg"]
        )
        graph = ViewGraph.from_view_set(vs, temporal_window=1)
        self.assertEqual(len(graph.nodes), 3)
        self.assertGreater(len(graph.edges), 0)
        neighbors = graph.get_local_neighbors("obs_1")
        self.assertIn("obs_2", neighbors)

    def test_camera_validation(self):
        k = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        rt = np.eye(4)[:3, :]
        res = CameraValidator.validate_camera_matrix(k, rt, (640, 480))
        self.assertTrue(res["valid"])

        rt_bad = rt.copy()
        rt_bad[0, 0] = -5.0
        res_bad = CameraValidator.validate_camera_matrix(k, rt_bad, (640, 480))
        self.assertFalse(res_bad["valid"])

    def test_dense_point_fusion_and_outliers(self):
        points = np.random.rand(100, 3)
        outliers = np.array([[100.0, 100.0, 100.0]])
        noisy_points = np.concatenate([points, outliers], axis=0)

        filtered = OutlierRejection.statistical_outlier_removal(noisy_points, nb_neighbors=5, std_ratio=1.5)
        self.assertLess(len(filtered), len(noisy_points))

    def test_fusion_orchestration(self):
        cam = CameraEstimate(
            camera_id="cam_1",
            intrinsics_k=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
            extrinsics_rt=np.eye(4)[:3, :],
            focal_lengths=(500.0, 500.0),
            principal_point=(320.0, 240.0),
            backend_name="test"
        )
        depth = np.ones((10, 10), dtype=np.float32) * 2.0
        fusion = DensePointFusion()
        fused = fusion.fuse([cam], [depth])
        self.assertIsInstance(fused, FusedWorldGeometry)

if __name__ == "__main__":
    unittest.main()
