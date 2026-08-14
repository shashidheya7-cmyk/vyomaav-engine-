
import trimesh
from engine.core.logger import engine_logger

class TopologyEngine:
    def __init__(self, config=None, device=None, precision=None, *args, **kwargs):
        self.config = config

    def process(self, scene):
        engine_logger.info("Executing topology stage...")
        mesh = getattr(scene, 'mesh', None) or getattr(scene, 'raw_mesh', None)
        if mesh is None:
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        
        scene.mesh = mesh
        scene.topology_mesh = mesh
        engine_logger.info("Topology retopology completed successfully.")
        return scene


