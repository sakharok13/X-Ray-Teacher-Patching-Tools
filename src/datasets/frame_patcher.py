import numpy as np

from abc import ABC, abstractmethod


class FramePatcher(ABC):
    """Patches a frame with new point cloud.
    """

    @property
    @abstractmethod
    def frame_id(self) -> str:
        """Returns processed frame id.

        :return: str
            ID of a frame, should be unique across a scene.
        """
        ...

    @property
    @abstractmethod
    def frame(self) -> np.ndarray:
        """Returns frame point cloud.

        :return: np.ndarray[float]
            Returns numpy array of kxm, where m is samples count.
        """
        ...

    @abstractmethod
    def patch_instance(self,
                       instance_id: str,
                       point_cloud: np.ndarray):
        """Replaces a point cloud of the instance in the frame with the given point cloud.

        :param instance_id: str
            Instance ID.
        :param point_cloud: np.ndarray[float]
            New point cloud.
        """
        ...
