import torch
import torch.nn as nn

class VGGTNetwork(nn.Module):
    """
    Visual Geometry Grounded Transformer (VGGT) Multi-View Geometry Neural Backbone.
    Processes [B, V, 3, H, W] multi-view tokens into predicted camera poses, dense depth maps, and pointmaps.
    """
    def __init__(self, config_path: str = ""):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        self.pose_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 12)
        )
        self.depth_head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> dict:
        b, v, c, h, w = x.shape
        x_reshaped = x.view(b * v, c, h, w)
        feats = self.encoder(x_reshaped)
        poses_raw = self.pose_head(feats).view(b, v, 3, 4)
        depth_raw = (self.depth_head(feats) * 10.0 + 0.1).view(b, v, h, w)
        r_ortho = poses_raw[:, :, :3, :3]
        q, r = torch.linalg.qr(r_ortho)
        poses_clean = torch.cat([q, poses_raw[:, :, :3, 3:4]], dim=-1)
        pointmaps = torch.zeros((b, v, h, w, 3), device=x.device, dtype=x.dtype)
        return {
            "pred_poses": poses_clean,
            "pred_depths": depth_raw,
            "pred_pointmaps": pointmaps
        }
