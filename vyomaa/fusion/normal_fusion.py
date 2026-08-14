import numpy as np

class NormalFusion:
    @staticmethod
    def compute_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
        if len(points) < 3:
            return np.zeros_like(points)
        from scipy.spatial import KDTree
        tree = KDTree(points)
        normals = np.zeros_like(points)
        for i, p in enumerate(points):
            _, idx = tree.query(p, k=min(k, len(points)))
            neighbors = points[idx]
            cov = np.cov(neighbors.T)
            _, eigenvecs = np.linalg.eigh(cov)
            normals[i] = eigenvecs[:, 0]
        return normals
