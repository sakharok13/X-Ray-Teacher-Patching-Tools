from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import Optional

from src.datasets.frame_patcher import FramePatcher


class Dataset(ABC):
    class SceneIterator(ABC):
        @abstractmethod
        def __next__(self):
            ...

    @property
    @abstractmethod
    def scenes(self) -> list:
        ...

    @abstractmethod
    def get_scene_iterator(self, scene_id: str) -> Dataset.SceneIterator:
        ...

    @abstractmethod
    def load_frame_patcher(self, scene_id: str, frame_id: str) -> FramePatcher:
        ...

    @abstractmethod
    def can_serialise_frame_point_cloud(self,
                                        scene_id: str,
                                        frame_id: str) -> bool:
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
        ...

    @abstractmethod
    def get_instance_point_cloud(self,
                                 scene_id: str,
                                 frame_id: str,
                                 instance_id: str,
                                 frame_point_cloud: np.ndarray) -> np.ndarray:
        ...
