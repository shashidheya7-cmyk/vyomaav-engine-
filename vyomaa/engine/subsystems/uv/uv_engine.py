
import trimesh
import numpy as np
from engine.core.logger import engine_logger

class UVEngine:
    def __init__(self, config=None, device=None, precision=None, *args, **kwargs):
        self.config = config

    def process(self, scene):
        engine_logger.info("Executing UV unwrapping stage...")
        mesh = getattr(scene, 'mesh', None)
        if mesh is not None and len(mesh.vertices) > 0:
            verts = mesh.vertices
            x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
            y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
            
            x_span = (x_max - x_min) if (x_max - x_min) > 1e-8 else 1.0
            y_span = (y_max - y_min) if (y_max - y_min) > 1e-8 else 1.0
            
            u = (verts[:, 0] - x_min) / x_span
            v = (verts[:, 1] - y_min) / y_span
            uvs = np.column_stack((u, v))
            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)

        scene.mesh = mesh
        engine_logger.info("UV unwrapping stage completed successfully.")
        return scene


