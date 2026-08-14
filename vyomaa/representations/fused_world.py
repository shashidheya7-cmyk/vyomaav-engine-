from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np

@dataclass
class DynamicCluster:
    object_id: str
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    source_views: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FusedWorldGeometry:
    fused_points: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    source_view_ids: List[str] = field(default_factory=list)
    object_ids: List[str] = field(default_factory=list)
    is_dynamic: Optional[np.ndarray] = None
    dynamic_clusters: Dict[str, DynamicCluster] = field(default_factory=dict)
    coordinate_frame: str = "world_opencv"
    scale_status: str = "up_to_scale"
    bounds: Dict[str, list[float]] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def export_ply(self, filepath: str, export_normals: bool = True) -> str:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        n_pts = len(self.fused_points)
        if n_pts == 0:
            return str(p)

        has_colors = self.colors is not None and len(self.colors) == n_pts
        has_normals = export_normals and self.normals is not None and len(self.normals) == n_pts
        has_conf = self.confidence is not None and len(self.confidence) == n_pts

        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {n_pts}",
            "property float x",
            "property float y",
            "property float z"
        ]
        if has_normals:
            header.extend(["property float nx", "property float ny", "property float nz"])
        if has_colors:
            header.extend(["property uchar red", "property uchar green", "property uchar blue"])
        if has_conf:
            header.append("property float confidence")
        header.append("end_header\n")

        lines = []
        for i in range(n_pts):
            pt = self.fused_points[i]
            parts = [f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}"]
            if has_normals:
                norm = self.normals[i]
                parts.append(f"{norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f}")
            if has_colors:
                c = self.colors[i]
                parts.append(f"{int(c[0])} {int(c[1])} {int(c[2])}")
            if has_conf:
                parts.append(f"{self.confidence[i]:.4f}")
            lines.append(" ".join(parts))

        with open(p, "w") as f:
            f.write("\n".join(header) + "\n".join(lines) + "\n")
        return str(p)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_points": len(self.fused_points),
            "static_points": int((~self.is_dynamic).sum()) if self.is_dynamic is not None else len(self.fused_points),
            "dynamic_points": int(self.is_dynamic.sum()) if self.is_dynamic is not None else 0,
            "dynamic_clusters_count": len(self.dynamic_clusters),
            "coordinate_frame": self.coordinate_frame,
            "scale_status": self.scale_status,
            "bounds": self.bounds,
            "provenance": self.provenance
        }
