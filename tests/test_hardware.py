"""Unit tests for GPUInfo discovery and VRAMManager allocation tracking."""

import unittest

from vyomaa.core.exceptions import HardwareError
from vyomaa.hardware.gpu_info import GPUInfo
from vyomaa.hardware.vram_manager import VRAMManager


class TestHardware(unittest.TestCase):

    def test_gpu_info_discovery(self):
        info = GPUInfo.discover()
        self.assertIsInstance(info.is_cuda_available, bool)
        self.assertIsInstance(info.total_vram_gb, float)

    def test_vram_manager_budget(self):
        mgr = VRAMManager(max_vram_budget_gb=50.0)
        status = mgr.get_status()
        self.assertEqual(status["budget_limit_gb"], 50.0)

        # Allocate 10 GB
        rec = mgr.register_allocation("TRELLIS", int(10.0 * (1024 ** 3)))
        self.assertEqual(rec.allocated_gb, 10.0)
        self.assertEqual(mgr.get_status()["managed_allocated_gb"], 10.0)

        # Allocate over budget -> raise HardwareError
        with self.assertRaises(HardwareError):
            mgr.register_allocation("HugeModel", int(45.0 * (1024 ** 3)))

        # Release
        mgr.release_allocation("TRELLIS")
        self.assertEqual(mgr.get_status()["managed_allocated_gb"], 0.0)


if __name__ == "__main__":
    unittest.main()
