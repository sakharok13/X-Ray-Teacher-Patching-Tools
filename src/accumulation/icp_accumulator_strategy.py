import numpy as np

from src.accumulation.accumulation_strategy import AccumulationStrategy
from src.utils.icp.icp import iterative_closest_point


class ICPAccumulatorStrategy(AccumulationStrategy):
    """Provides a strategy that concatenates point clouds 'as is'.

    The strategy assumes that the point clouds are already aligned
    and no further transformation is required.
    """

    def on_merge(self,
                 initial_point_cloud: np.ndarray,
                 next_point_cloud: np.ndarray,
                 frame_no: int) -> np.ndarray:
        if next_point_cloud.size == 0:
            return initial_point_cloud
        elif initial_point_cloud.size == 0:
            return next_point_cloud
        else:
            aligned_next_point_cloud = iterative_closest_point(
                source_points=next_point_cloud,
                destination_points=initial_point_cloud,
            )
            return np.concatenate((initial_point_cloud, aligned_next_point_cloud), axis=1)
