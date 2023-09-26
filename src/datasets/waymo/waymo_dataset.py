import os
import sys
import open3d as o3d
import numpy as np

from typing import Optional
from functools import lru_cache

from src.datasets.dataset import Dataset
from src.datasets.frame_patcher import FramePatcher
from src.datasets.waymo.waymo_scene_iterator import WaymoSceneIterator
from src.datasets.waymo.waymo_utils import find_all_scenes, load_frame_descriptors, load_frame_point_cloud



class WaymoDataset(Dataset):
    def __init__(self,
                 dataset_root: str):
        self.__dataset_root = dataset_root
        self.__scene_ids = find_all_scenes(dataset_root=dataset_root)

    @property
    def scenes(self) -> list:
        return list(self.__scene_ids)

    def get_scene_iterator(self, scene_id: str) -> Dataset.SceneIterator:
        scene_descriptors = self.__load_scene_descriptors(scene_id=scene_id)
        return WaymoSceneIterator(scene_id=scene_id, scene_descriptors=scene_descriptors)

    def load_frame_patcher(self, frame_id: str) -> FramePatcher:
        pass

    def serialise_frame_point_clouds(self,
                                     frame_id: str,
                                     frame_point_cloud: np.ndarray) -> Optional[str]:
        pass

    def get_frame_point_cloud(self,
                              frame_id: str) -> np.ndarray:
        pass
        # return load_frame_point_cloud(dataset_root=self.__dataset_root,
        #                               scene_id=)

    def get_instance_point_cloud(self,
                                 frame_id: str,
                                 instance_id: str,
                                 frame_point_cloud: np.ndarray) -> np.ndarray:
        pass

    @lru_cache(maxsize=12)
    def __load_scene_descriptors(self,
                                 scene_id: str) -> list:
        return load_frame_descriptors(dataset_root=self.__dataset_root,
                                      scene_id=scene_id)
