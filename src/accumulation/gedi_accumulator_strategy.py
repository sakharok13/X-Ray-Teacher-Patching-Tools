import numpy as np
import open3d as o3d
from src.accumulation.accumulation_strategy import AccumulationStrategy
from src.utils.gedi_registration import run_point_cloud_registration_o3d


class GediAccumulatorStrategy(AccumulationStrategy):
    """Provides a strategy that concatenates point clouds 'as is'.

    The strategy assumes that the point clouds are already aligned
    and no further transformation is required.
    """

    def on_merge(self,
                 initial_point_cloud: np.ndarray[float],
                 next_point_cloud: np.ndarray[float]) -> np.ndarray[float]:
        return np.concatenate((initial_point_cloud, next_point_cloud), axis=1)

    def on_registration(
            self,
            initial_point_cloud: o3d.geometry.PointCloud,
            next_point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        size = len(next_point_cloud.points)
        return run_point_cloud_registration_o3d(
            initial_point_cloud, next_point_cloud, size)
