import numpy as np

from abc import ABC, abstractmethod


class FramePatcher(ABC):

    @property
    @abstractmethod
    def frame(self) -> np.ndarray[float]:
        ...

    @abstractmethod
    def patch_instance(self,
                       instance_id: str,
                       point_cloud: np.ndarray[float]) -> bool:
        ...
