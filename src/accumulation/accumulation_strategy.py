import numpy as np

from abc import ABC, abstractmethod


class AccumulationStrategy(ABC):
    """Provides a strategy to merge cloud points.
    """

    @abstractmethod
    def on_merge(self,
                 initial_point_cloud: np.ndarray[float],
                 next_point_cloud: np.ndarray[float],
                 frame_id: int) -> np.ndarray[float]:
        """Merges two point clouds into a single cloud.

        :param initial_point_cloud: np.ndarray[float]
            Main point cloud.
        :param next_point_cloud: np.ndarray[float]
            Point cloud that we are trying to append to the main one.
        :param frame_id: int
        :return: np.ndarray[float]
            Merged point cloud.
        """
        ...
