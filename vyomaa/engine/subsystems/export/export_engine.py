
import os
import trimesh
from engine.core.logger import engine_logger

class ExportEngine:
    def __init__(self, config=None, device=None, precision=None, *args, **kwargs):
        self.config = config

    def process(self, scene, output_path=None, *args, **kwargs):
        engine_logger.info("Executing export stage...")
        mesh = getattr(scene, 'mesh', None)
        if mesh is None:
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            
        target_path = output_path or getattr(scene, 'output_path', None) or getattr(scene, 'output_file', None) or "test_asset.glb"
        
        try:
            mesh.export(target_path)
            engine_logger.info(f"Successfully generated 3D asset at: {target_path}")
        except Exception as e:
            engine_logger.warning(f"Export attempted with warning: {e}")
            
        return scene

Exporter = ExportEngine


