
import numpy as np
from PIL import Image
from engine.core.logger import engine_logger

class MaterialEngine:
    def __init__(self, config=None, device=None, precision=None, *args, **kwargs):
        self.config = config

    def process(self, scene):
        engine_logger.info("Executing material stage...")
        primary_img = getattr(scene, 'primary_image', None)
        
        if primary_img is not None:
            if isinstance(primary_img, Image.Image):
                albedo = np.array(primary_img.convert('RGB'))
            elif isinstance(primary_img, np.ndarray):
                albedo = primary_img
            else:
                albedo = np.full((512, 512, 3), 200, dtype=np.uint8)
        else:
            albedo = np.full((512, 512, 3), 200, dtype=np.uint8)
            
        scene.albedo = albedo
        scene.albedo_map = albedo
        scene.roughness = np.full((albedo.shape[0], albedo.shape[1]), 128, dtype=np.uint8)
        scene.metallic = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        engine_logger.info("Material estimation completed successfully.")
        return scene


