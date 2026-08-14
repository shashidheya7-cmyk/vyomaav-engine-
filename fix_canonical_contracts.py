import re
import subprocess
import sys

# Read vyomaa/multiview/contracts.py
with open("vyomaa/multiview/contracts.py", "r") as f:
    content = f.read()

# Replace any local class FusedWorldGeometry definition with the canonical import
# Pattern to remove class FusedWorldGeometry block if defined in contracts.py
content = re.sub(r'class FusedWorldGeometry[\s\S]*?(?=\nclass |\Z)', '', content)

# Ensure the import from representations is cleanly placed at top
if "from vyomaa.representations.fused_world import FusedWorldGeometry, DynamicCluster" not in content:
    content = "from vyomaa.representations.fused_world import FusedWorldGeometry, DynamicCluster\n" + content
else:
    # Clean duplicate import lines if any
    content = "from vyomaa.representations.fused_world import FusedWorldGeometry, DynamicCluster\n" + "\n".join(
        line for line in content.splitlines() if "from vyomaa.representations.fused_world import" not in line
    )

with open("vyomaa/multiview/contracts.py", "w") as f:
    f.write(content.strip() + "\n")

print("[✓] Replaced inline duplicate FusedWorldGeometry with canonical import in vyomaa/multiview/contracts.py")

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
