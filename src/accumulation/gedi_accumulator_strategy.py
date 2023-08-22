import numpy as np
from src.accumulation.accumulation_strategy import AccumulationStrategy


class GediAccumulatorStrategy(AccumulationStrategy):
    """Provides a strategy that concatenates point clouds 'as is'.

    The strategy assumes that the point clouds are already aligned
    and no further transformation is required.
    """

    def on_merge(self,
                 initial_point_cloud: np.ndarray[float],
                 next_point_cloud: np.ndarray[float]) -> np.ndarray[float]:
        return np.concatenate((initial_point_cloud, next_point_cloud), axis=1)