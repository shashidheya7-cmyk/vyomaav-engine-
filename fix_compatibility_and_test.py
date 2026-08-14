from pathlib import Path
import subprocess
import sys

# Append backward-compatibility wrapper to vyomaa/fusion/dense_point_fusion.py
compat_code = '''

class DensePointFusion(DenseWorldFusionEngine):
    """Backward-compatibility wrapper for Phase 4B contracts."""
    def fuse(self, cameras, depth_maps, confidences=None, colors=None):
        from vyomaa.multiview.contracts import ViewSet
        obs_ids = [c.camera_id for c in cameras]
        vs = ViewSet(observation_ids=obs_ids, timestamps=[float(i) for i in range(len(cameras))], keyframe_flags=[True]*len(cameras), image_paths=[])
        return self.fuse_multiview(
            view_set=vs,
            cameras=cameras,
            vggt_depths=depth_maps,
            images_rgb=colors
        )
'''

with open("vyomaa/fusion/dense_point_fusion.py", "a") as f:
    f.write(compat_code)

print("[✓] Appended DensePointFusion compatibility class to vyomaa/fusion/dense_point_fusion.py")

# Run full test suite
cmd = [
    sys.executable, "-m", "unittest",
    "tests/test_phase4c_fusion.py",
    "tests/test_vggt_real_gpu.py",
    "tests/test_sam2_multiframe.py",
    "tests/test_sam2_real_gpu.py",
    "tests/test_sam2_worker.py",
    "tests/test_multiview_foundation.py"
]
res = subprocess.run(cmd)
sys.exit(res.returncode)
