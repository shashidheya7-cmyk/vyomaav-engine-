import numpy as np

class OutlierRejection:
    @staticmethod
    def statistical_outlier_removal(points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
        if len(points) == 0:
            return points
        from scipy.spatial import KDTree
        tree = KDTree(points)
        distances, _ = tree.query(points, k=min(nb_neighbors + 1, len(points)))
        mean_distances = np.mean(distances[:, 1:], axis=1) if distances.shape[1] > 1 else np.zeros(len(points))
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        threshold = global_mean + std_ratio * global_std
        mask = mean_distances < threshold
        return points[mask]
