import subprocess
import sys

# 1. Update vyomaa/multiview/contracts.py to reference canonical FusedWorldGeometry
with open("vyomaa/multiview/contracts.py", "r") as f:
    code = f.read()

# Ensure canonical import is present at the top
if "from vyomaa.representations.fused_world import FusedWorldGeometry" not in code:
    # If class FusedWorldGeometry was defined inline, alias it to the canonical representation
    patch_import = "from vyomaa.representations.fused_world import FusedWorldGeometry, DynamicCluster\n"
    code = patch_import + code

with open("vyomaa/multiview/contracts.py", "w") as f:
    f.write(code)

print("[✓] Unified FusedWorldGeometry contract across vyomaa.multiview.contracts")

# 2. Run full test suite
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
