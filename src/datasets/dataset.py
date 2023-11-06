from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, Iterable

from src.datasets.frame_patcher import FramePatcher


class Dataset(ABC):
    class SceneIterator(ABC, Iterable):
        """Iterates through all frames in a scene.

        The complexity of __init__ method should not
        exceed O(1).
        """

        @abstractmethod
        def __next__(self):
            """Retrieves information about the next frame.

            Runtime complexity of getting the next frame information is O(1).

            :raises:
                StopIteration when there are no elements left.
            :return:
                A tuple of frame_id (string) and associated frame meta-information,
                grouped in FrameDescriptor class.
            """
            ...

    @property
    @abstractmethod
    def dataroot(self) -> str:
        ...

    @property
    @abstractmethod
    def scenes(self) -> list:
        ...

    @abstractmethod
    def get_scene_iterator(self, scene_id: str) -> Dataset.SceneIterator:
        """Returns a scene iterator for the given scene_id.

        Scene iterator navigates through all available frames.

        Creating a scene iterator is a lightweight operation:
        performance of the method should not exceed O(1).

        :param scene_id: str
            Unique scene identifier.
        :return:
            An instance of SceneIterator.
        """
        ...

    @abstractmethod
    def load_frame_patcher(self, scene_id: str, frame_id: str) -> FramePatcher:
        ...

    @abstractmethod
    def can_serialise_frame_point_cloud(self,
                                        scene_id: str,
                                        frame_id: str) -> bool:
        """Checks whether it is possible to serialise a point cloud for the given frame in the scene.

        It is possible to serialise a point cloud for the frame if there is no
        serialised version of the same point cloud on disk.

        Runtime complexity is O(1).

        :param scene_id: str
            Unique scene identifier.
        :param frame_id: str
            Unique frame identifier.
        :return:
            True if it is possible to serialise the point cloud and
            False otherwise.
        """
        ...

    @abstractmethod
    def serialise_frame_point_clouds(self,
                                     scene_id: str,
                                     frame_id: str,
                                     frame_point_cloud: np.ndarray) -> Optional[str]:
        ...

    @abstractmethod
    def get_frame_point_cloud(self,
                              scene_id: str,
                              frame_id: str) -> np.ndarray:
        """Loads frame point cloud.

        Usually, loads the point cloud in memory.
        Potentially, extremely heavy operation.
        Runtime consideration is at least O(N*d),
        where N is the number of point in the point cloud
        and D is their dimensionality.

        It is reasonable to expect N >= 1000 and d >= 3.
        """
        ...

    @abstractmethod
    def get_instance_point_cloud(self,
                                 scene_id: str,
                                 frame_id: str,
                                 instance_id: str,
                                 frame_point_cloud: np.ndarray) -> np.ndarray:
        """Returns point cloud of the specified instance.

        Runtime complexity is O(N*d).

        """
        ...
