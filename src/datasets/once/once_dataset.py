from __future__ import annotations

import os
import os.path as osp
import numpy as np

from typing import Optional

from src.datasets.dataset import Dataset
from src.datasets.frame_patcher import FramePatcher
from src.datasets.once.once_scene_iterator import OnceSceneIterator
from src.datasets.once.once_frame_patcher import OnceFramePatcher
from src.datasets.once.once_utils import ONCE, get_frame_point_cloud, get_instance_point_cloud


class OnceDataset(Dataset):
    def __init__(self,
                 split='train',
                 dataroot='./temp/ONCE'):
        self.__once = ONCE(dataroot, split)
        self.__once._collect_basic_infos(split)
        self.__dataroot = dataroot
        self.__scenes = os.listdir(osp.join(self.__dataroot, 'data'))   # ['000013']
        self.__scenes_lookup = {str(i): scene for i, scene in enumerate(self.__scenes)}

    @property
    def scenes(self) -> list:
        return self.__scenes

    def get_scene_iterator(self, scene_id: str) -> Dataset.SceneIterator:
        assert scene_id in self.__scenes_lookup

        return OnceSceneIterator(scene=self.__scenes_lookup[scene_id],
                                 once=self.__once)

    def load_frame_patcher(self, frame_id: str,
                                 scene_id: str) -> FramePatcher:

        return OnceFramePatcher.load(seq_id=self.__scenes_lookup[scene_id],
                                     frame_id=frame_id,
                                     once=self.__once)

    def serialise_frame_point_clouds(self,
                                     scene_id: str,
                                     frame_id: str,
                                     frame_point_cloud: np.ndarray) -> Optional[str]:
        path_to_save = self.__get_patched_folder_and_filename(self.__scenes_lookup[scene_id], frame_id)

        dir_path = os.path.dirname(path_to_save)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        OnceFramePatcher.serialise(path=path_to_save,
                                   point_cloud=frame_point_cloud)

        return path_to_save

    def can_serialise_frame_point_cloud(self,
                                        scene_id: str,
                                        frame_id: str) ->bool:
        path_to_save = self.__get_patched_folder_and_filename(self.__scenes_lookup[scene_id], frame_id)

        # We can serialise point cloud if there is no point cloud saved.
        return not os.path.exists(path_to_save)

    def dataroot(self):
        return self.__dataroot

    def get_frame_point_cloud(self,
                              scene_id: str,
                              frame_id: str) -> np.ndarray:

        return get_frame_point_cloud(seq_id=self.__scenes_lookup[scene_id],
                                     frame_id=frame_id,
                                     once=self.__once)

    def get_instance_point_cloud(self,
                                 scene_id: str,
                                 frame_id: str,
                                 instance_id: str,
                                 frame_point_cloud: np.ndarray) -> np.ndarray:
        return get_instance_point_cloud(seq_id=self.__scenes_lookup[scene_id],
                                        frame_id=frame_id,
                                        instance_id=instance_id,
                                        frame_point_cloud=frame_point_cloud,
                                        once=self.__once)

    def __get_patched_folder_and_filename(self, scene_id: str, frame_id: str):
        folder_name = f"{scene_id}_patched"
        file_name = f"{frame_id}.bin"

        return os.path.join(self.__dataroot, 'data', folder_name, 'lidar_roof', file_name)
