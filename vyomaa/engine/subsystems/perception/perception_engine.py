
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from engine.core.logger import engine_logger
from engine.core.exceptions import PerceptionError

class PerceptionEngine:
    def __init__(self, config=None, device=None, precision=None, *args, **kwargs):
        self.config = config if isinstance(config, dict) else getattr(config, '__dict__', {})
        self.device_str = str(device) if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device_str)
        self.precision = precision
        
        if isinstance(self.config, dict):
            self.model_id = self.config.get('model_id', 'ZhengPeng7/BiRefNet')
        else:
            self.model_id = getattr(config, 'model_id', 'ZhengPeng7/BiRefNet')
            
        self.model = None

    def initialize(self):
        try:
            from transformers import AutoModelForImageSegmentation
            engine_logger.info(f"Loading perception model: {self.model_id}")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.model_id, trust_remote_code=True
            ).to(self.device).eval()
            engine_logger.info("Perception model loaded successfully.")
        except Exception as e:
            engine_logger.warning(f"Could not load {self.model_id} via AutoModelForImageSegmentation ({e}). Using standard alpha mask fallback.")
            self.model = None

    def process(self, scene):
        engine_logger.info("Executing perception stage...")
        if self.model is None:
            self.initialize()

        raw_img = None
        
        # Search all attributes on scene dynamically
        for attr in dir(scene):
            if attr.startswith('_'): continue
            try:
                val = getattr(scene, attr)
                if isinstance(val, Image.Image):
                    raw_img = val.convert('RGB')
                    break
                elif isinstance(val, str) and os.path.exists(val) and val.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    raw_img = Image.open(val).convert('RGB')
                    break
                elif isinstance(val, torch.Tensor):
                    arr = val.cpu().numpy()
                    if arr.shape[0] in [1, 3, 4]:
                        arr = np.transpose(arr, (1, 2, 0))
                    raw_img = Image.fromarray((arr * 255).astype(np.uint8)).convert('RGB')
                    break
            except Exception:
                pass

        # Fallback local search
        if raw_img is None:
            for fallback in ['image.png', 'my_input.png', 'input.png', 'test.png']:
                if os.path.exists(fallback):
                    raw_img = Image.open(fallback).convert('RGB')
                    break

        if raw_img is None:
            raise PerceptionError("No valid input image provided to PerceptionEngine.")

        if self.model is not None:
            try:
                transform = transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                input_tensor = transform(raw_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    preds = self.model(input_tensor)[-1].sigmoid().cpu()
                    mask_tensor = preds[0].squeeze()
                    mask = transforms.ToPILImage()(mask_tensor).resize(raw_img.size)
                
                rgba_img = raw_img.copy()
                rgba_img.putalpha(mask)
                scene.primary_image = rgba_img
                engine_logger.info("Background removal completed via BiRefNet.")
                return scene
            except Exception as e:
                engine_logger.warning(f"BiRefNet inference failed ({e}). Falling back to RGBA image pass-through.")

        rgba_img = raw_img.convert('RGBA')
        scene.primary_image = rgba_img
        engine_logger.info("Perception stage completed with RGBA conversion.")
        return scene

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


