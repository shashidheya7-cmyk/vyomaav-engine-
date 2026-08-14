
import trimesh
import numpy as np
from PIL import Image
from engine.core.logger import engine_logger

class GeometryEngine:
    def __init__(self, config=None, device=None, precision=None, *args, **kwargs):
        self.config = config

    def process(self, scene):
        engine_logger.info("Executing geometry stage...")
        
        primary_img = getattr(scene, 'primary_image', None)
        if primary_img is None or not isinstance(primary_img, Image.Image):
            primary_img = Image.new('RGBA', (128, 128), (200, 200, 200, 255))

        img_rgba = primary_img.convert('RGBA').resize((128, 128))
        img_np = np.array(img_rgba)
        
        rgb = img_np[:, :, :3]
        alpha = img_np[:, :, 3] / 255.0
        gray = np.mean(rgb, axis=2)

        # Key out transparent background or solid dark/light backgrounds
        if (np.max(alpha) - np.min(alpha)) > 0.2:
            mask = alpha > 0.2
        else:
            mask = (gray > 25) & (gray < 245)

        if not np.any(mask):
            mask = np.ones((128, 128), dtype=bool)

        heightmap = gray / 255.0
        res_y, res_x = 128, 128
        x = np.linspace(-1, 1, res_x)
        y = np.linspace(-1, 1, res_y)
        xx, yy = np.meshgrid(x, y)

        # Build index mapping table ONLY for active foreground pixels
        grid_idx = np.full((res_y, res_x), -1, dtype=int)
        valid_coords = np.argwhere(mask)
        
        v_front, v_back = [], []
        for idx, (r, c) in enumerate(valid_coords):
            grid_idx[r, c] = idx
            px = xx[r, c]
            py = -yy[r, c]
            pz = heightmap[r, c] * 0.35 + 0.05
            
            v_front.append([px, py, pz])
            v_back.append([px, py, -pz])

        v_front = np.array(v_front)
        v_back = np.array(v_back)

        faces = []
        # Connect faces only between adjacent valid burger pixels
        for r in range(res_y - 1):
            for c in range(res_x - 1):
                i00 = grid_idx[r, c]
                i10 = grid_idx[r, c + 1]
                i01 = grid_idx[r + 1, c]
                i11 = grid_idx[r + 1, c + 1]

                if i00 != -1 and i10 != -1 and i01 != -1:
                    faces.append([i00, i10, i01])
                if i10 != -1 and i11 != -1 and i01 != -1:
                    faces.append([i10, i11, i01])

        if len(faces) > 0:
            f_arr = np.array(faces)
            num_v = len(v_front)
            f_back = f_arr[:, ::-1] + num_v
            
            all_verts = np.vstack([v_front, v_back])
            all_faces = np.vstack([f_arr, f_back])
            
            mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=True)
        else:
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

        scene.mesh = mesh
        scene.raw_mesh = mesh
        engine_logger.info(f"Geometry stage completed cleanly with {len(mesh.faces)} faces.")
        return scene


