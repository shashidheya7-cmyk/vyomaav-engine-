import numpy as np
from typing import Dict, Any

class DepthCrossCheck:
    @staticmethod
    def evaluate(depth_vggt: np.ndarray, depth_da: np.ndarray, threshold: float = 0.25) -> Dict[str, Any]:
        valid = (depth_vggt > 0) & (depth_da > 0) & np.isfinite(depth_vggt) & np.isfinite(depth_da)
        if not valid.any():
            return {
                "valid_overlap_ratio": 0.0,
                "median_ratio": 1.0,
                "correlation": 0.0,
                "disagreement_percentage": 100.0,
                "status": "NO_VALID_OVERLAP"
            }

        vggt_vals = depth_vggt[valid].astype(np.float64)
        da_vals = depth_da[valid].astype(np.float64)

        ratios = vggt_vals / np.maximum(da_vals, 1e-6)
        median_ratio = float(np.median(ratios))

        vggt_norm = (vggt_vals - np.mean(vggt_vals)) / (np.std(vggt_vals) + 1e-6)
        da_norm = (da_vals - np.mean(da_vals)) / (np.std(da_vals) + 1e-6)
        corr = float(np.clip(np.mean(vggt_norm * da_norm), -1.0, 1.0))

        da_scaled = da_vals * median_ratio
        relative_diff = np.abs(vggt_vals - da_scaled) / np.maximum(vggt_vals, 1e-3)
        disagreement_pct = float(np.mean(relative_diff > threshold) * 100.0)

        return {
            "valid_overlap_ratio": float(np.mean(valid)),
            "median_ratio": median_ratio,
            "correlation": corr,
            "disagreement_percentage": disagreement_pct,
            "status": "CROSS_CHECK_VALIDATED"
        }
