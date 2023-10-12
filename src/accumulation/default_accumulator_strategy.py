import numpy as np

from src.accumulation.accumulation_strategy import AccumulationStrategy


class DefaultAccumulatorStrategy(AccumulationStrategy):
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
        else:
            return np.concatenate((initial_point_cloud, next_point_cloud), axis=1)
