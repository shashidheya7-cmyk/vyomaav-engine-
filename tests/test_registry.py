"""Unit tests for Model and Task Registries."""

import unittest

from vyomaa.core.exceptions import RegistryError
from vyomaa.core.registry import Registry, ModelSpec, MODEL_REGISTRY
from vyomaa.core.types import ModelCapability


class TestRegistry(unittest.TestCase):

    def test_registry_registration_and_lookup(self):
        reg = Registry("TEST_REG")
        spec = ModelSpec(name="TestWorker", capability=ModelCapability.MONOCULAR_DEPTH)

        @reg.register("TestWorker", spec=spec)
        class DummyWorker:
            pass

        self.assertIn("TestWorker", reg.list_all())
        self.assertEqual(reg.get("TestWorker"), DummyWorker)
        self.assertEqual(reg.get_spec("TestWorker").capability, ModelCapability.MONOCULAR_DEPTH)

    def test_duplicate_registration_raises(self):
        reg = Registry("TEST_DUP")
        reg.register("ModuleA")(lambda: 1)
        with self.assertRaises(RegistryError):
            reg.register("ModuleA")(lambda: 2)


if __name__ == "__main__":
    unittest.main()
