
"""Nine-stage state-driven VYOMAA pipeline orchestration."""

from __future__ import annotations

import time

from ..core.config import EngineConfig
from ..core.exceptions import PipelineExecutionError
from ..core.logger import get_logger
from ..scene.scene import Scene
from ..subsystems.export.export_engine import ExportEngine
from ..subsystems.fusion.fusion_engine import FusionEngine
from ..subsystems.geometry.geometry_engine import GeometryEngine
from ..subsystems.material.material_engine import MaterialEngine
from ..subsystems.multiview.multiview_engine import MultiViewEngine
from ..subsystems.perception.perception_engine import PerceptionEngine
from ..subsystems.texture.texture_engine import TextureEngine
from ..subsystems.topology.topology_engine import TopologyEngine
from ..subsystems.uv.uv_engine import UVEngine


class Orchestrator:
    """Execute ordered state transformations from image ingestion through GLB export."""

    _ORDER = ("perception", "multiview", "fusion", "geometry", "topology", "uv", "texture", "material", "export")

    def __init__(self, config: EngineConfig) -> None:
        self.config, self.logger = config, get_logger()
        self.perception_engine = PerceptionEngine(config.perception, config.device, config.precision)
        self.multiview_engine = MultiViewEngine(config.multiview, config.device, config.precision)
        self.fusion_engine = FusionEngine({**config.fusion, "device": config.device})
        self.geometry_engine = GeometryEngine(config.geometry)
        self.topology_engine = TopologyEngine(config.topology)
        self.uv_engine = UVEngine(config.uv)
        self.texture_engine = TextureEngine(config.texture, config.device)
        self.material_engine = MaterialEngine(config.material)
        self.export_engine = ExportEngine(config.export)

    def run(self, scene: Scene) -> Scene:
        """Run each enabled stage in canonical order, with timing diagnostics."""
        selected = set(self.config.pipeline_stages)
        unknown = selected.difference(self._ORDER)
        missing = set(self._ORDER).difference(selected)
        if unknown or missing:
            raise PipelineExecutionError(f"a complete production run requires all nine stages; unknown={sorted(unknown)}, missing={sorted(missing)}")
        operations = {
            "perception": self.perception_engine.process,
            "multiview": self.multiview_engine.process,
            "fusion": self.fusion_engine.process,
            "geometry": self.geometry_engine.process,
            "topology": self.topology_engine.process,
            "uv": self.uv_engine.process,
            "texture": self.texture_engine.process,
            "material": self.material_engine.process,
            "export": self.export_engine.process,
        }
        try:
            for stage in self._ORDER:
                self.logger.info("Executing %s stage", stage)
                started = time.perf_counter()
                scene = operations[stage](scene)
                scene.diagnostics.setdefault("stage_seconds", {})[stage] = round(time.perf_counter() - started, 3)
            self.logger.info("Pipeline exported GLB to %s", scene.output_path)
            return scene
        except PipelineExecutionError:
            raise
        except Exception as exc:
            raise PipelineExecutionError(f"pipeline failed: {exc}") from exc


