import numpy as np

from src.accumulation.accumulation_strategy import AccumulationStrategy


class NoOpAccumulatorStrategy(AccumulationStrategy):
    """No-op strategy, always return the initial point cloud.
    """

    def on_merge(self,
                 initial_point_cloud: np.ndarray,
                 next_point_cloud: np.ndarray,
                 frame_no: int) -> np.ndarray:
        return initial_point_cloud
