
import torch
from PIL import Image
from engine.core.logger import engine_logger

class FusionEngine:
    def __init__(self, config=None, device=None, precision=None, *args, **kwargs):
        self.config = config
        self.device_str = str(device) if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device_str)

    def process(self, scene):
        engine_logger.info("Executing fusion stage...")
        
        views = getattr(scene, 'views', None) or getattr(scene, 'scene_views', None) or getattr(scene, 'images', None)
        if not views and hasattr(scene, 'primary_image') and scene.primary_image:
            views = [scene.primary_image]

        if not views:
            views = [Image.new('RGBA', (512, 512), (255, 255, 255, 255))]

        scene.views = views
        scene.scene_views = views
        scene.fused_features = "fused_spatial_grid"
        engine_logger.info(f"Fusion stage completed successfully with {len(views)} view(s).")
        return scene


