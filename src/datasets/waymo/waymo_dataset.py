import os
import sys
import open3d as o3d
import numpy as np

from typing import Optional
from functools import lru_cache

from src.datasets.dataset import Dataset
from src.datasets.frame_patcher import FramePatcher
from src.datasets.waymo.waymo_frame_patcher import WaymoFramePatcher
from src.datasets.waymo.waymo_scene_iterator import WaymoSceneIterator
from src.datasets.waymo.waymo_utils import find_all_scenes, load_scene_descriptor, get_frame_point_cloud, get_instance_point_cloud, get_frame_index


class WaymoDataset(Dataset):
    def __init__(self,
                 dataset_root: str):
        self.__dataset_root = dataset_root
        self.__scene_ids = find_all_scenes(dataset_root=dataset_root)

    @property
    def scenes(self) -> list:
        return list(self.__scene_ids)

    def get_scene_iterator(self, scene_id: str) -> Dataset.SceneIterator:
        scene_descriptor = self.__load_scene_descriptor(scene_id=scene_id)
        return WaymoSceneIterator(scene_id=scene_id, scene_descriptor=scene_descriptor)

    def load_frame_patcher(self,
                           scene_id: str,
                           frame_id: str) -> FramePatcher:
        scene_descriptor = self.__load_scene_descriptor(scene_id=scene_id)

        return WaymoFramePatcher.load(dataset_root=self.__dataset_root,
                                      scene_id=scene_id,
                                      frame_id=frame_id,
                                      scene_descriptor=scene_descriptor)

    def serialise_frame_point_clouds(self,
                                     scene_id: str,
                                     frame_id: str,
                                     frame_point_cloud: np.ndarray) -> Optional[str]:
        scene_descriptor = self.__load_scene_descriptor(scene_id=scene_id)
        frame_descriptor = scene_descriptor[frame_id]

        patched_root_folder = os.path.join(self.__dataset_root, 'patched')
        os.makedirs(patched_root_folder, exist_ok=True)

        patched_scene_folder = os.path.join(patched_root_folder, scene_id)
        os.makedirs(patched_scene_folder, exist_ok=True)

        frame_index = get_frame_index(frame_descriptor)
        path_to_save = os.path.join(patched_scene_folder, f"{frame_index:04d}.npy")

        WaymoFramePatcher.serialise(path=path_to_save,
                                    point_cloud=frame_point_cloud)

        return path_to_save

    def get_frame_point_cloud(self,
                              scene_id: str,
                              frame_id: str) -> np.ndarray:
        scene_descriptor = self.__load_scene_descriptor(scene_id=scene_id)
        frame_descriptor = scene_descriptor[frame_id]

        return get_frame_point_cloud(dataset_root=self.__dataset_root,
                                     scene_id=scene_id,
                                     frame_descriptor=frame_descriptor)

    def get_instance_point_cloud(self,
                                 scene_id: str,
                                 frame_id: str,
                                 instance_id: str,
                                 frame_point_cloud: np.ndarray) -> np.ndarray:
        scene_descriptor = self.__load_scene_descriptor(scene_id=scene_id)
        return get_instance_point_cloud(frame_point_cloud=frame_point_cloud,
                                        instance_id=instance_id,
                                        frame_descriptor=scene_descriptor[frame_id])

    @lru_cache(maxsize=12)
    def __load_scene_descriptor(self,
                                scene_id: str) -> dict:
        return load_scene_descriptor(dataset_root=self.__dataset_root,
                                     scene_id=scene_id)
