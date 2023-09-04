import numpy as np
import open3d as o3d
from src.accumulation.accumulation_strategy import AccumulationStrategy
from src.utils.gedi_registration import run_point_cloud_registration_o3d
from src.utils.o3d_helper import convert_to_o3d_pointcloud, convert_to_numpy_array

class GediAccumulatorStrategy(AccumulationStrategy):
    """Provides a strategy that concatenates point clouds 'as is'.

    The strategy assumes that the point clouds are already aligned
    and no further transformation is required.
    """

    def on_merge(self,
                 initial_point_cloud: np.ndarray[float],
                 next_point_cloud: np.ndarray[float]) -> np.ndarray[float]:
        size_init = initial_point_cloud.shape[1]
        size_next = next_point_cloud.shape[1]


        if size_init > 30 and size_next > 30:
            initial_point_cloud_o3d = convert_to_o3d_pointcloud(initial_point_cloud.T)
            next_point_cloud_o3d = convert_to_o3d_pointcloud(next_point_cloud.T)

            result_point_cloud_o3d = run_point_cloud_registration_o3d(
                next_point_cloud_o3d, initial_point_cloud_o3d, size_init, size_next)

            result_point_cloud = convert_to_numpy_array(result_point_cloud_o3d)
            return result_point_cloud

        else:
            result_point_cloud = np.concatenate((initial_point_cloud, next_point_cloud), axis=1)
            return result_point_cloud





