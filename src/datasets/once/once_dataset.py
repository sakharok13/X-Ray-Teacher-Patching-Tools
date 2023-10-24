from __future__ import annotations

import os
import os.path as osp
import numpy as np

from typing import Optional

from src.datasets.dataset import Dataset
from src.datasets.frame_patcher import FramePatcher
from src.datasets.once.once_scene_iterator import OnceSceneIterator
from src.datasets.once.once_frame_patcher import OnceFramePatcher
from src.datasets.once.once_utils import ONCE, get_instance_point_cloud


class OnceDataset(Dataset):
    """
    dataset structure:
    - data_root
        -ImageSets
            - train_split.txt
            - val_split.txt
            - test_split.txt
            - raw_split.txt
        - data
            - seq_id
                - cam01
                - cam03
                - ...
                -
    """

    def __init__(self,
                 dataset_root: str,
                 split: str,
                 overwrite: bool):
        self.__dataset_root = dataset_root
        self.__overwrite = overwrite
        self.__scenes_root = os.path.join(self.__dataset_root, 'data')

        self.__once = ONCE(self.__dataset_root, self.__scenes_root, split)
        self.__scene_ids = self.__once.get_scenes_in_split(split)

    @property
    def dataroot(self) -> str:
        return self.__dataset_root

    @property
    def scenes(self) -> list:
        return list(self.__scene_ids)

    def get_scene_iterator(self, scene_id: str) -> Dataset.SceneIterator:
        assert scene_id in self.__scene_ids, \
            f"Unknown scene id {scene_id}"

        return OnceSceneIterator(scene_id=scene_id,
                                 once=self.__once)

    def load_frame_patcher(self,
                           scene_id: str,
                           frame_id: str) -> FramePatcher:
        assert scene_id in self.__scene_ids, \
            f"Unknown scene id {scene_id}"

        return OnceFramePatcher.load(scene_id=scene_id,
                                     frame_id=frame_id,
                                     once=self.__once)

    def serialise_frame_point_clouds(self,
                                     scene_id: str,
                                     frame_id: str,
                                     frame_point_cloud: np.ndarray) -> Optional[str]:
        assert scene_id in self.__scene_ids, \
            f"Unknown scene id {scene_id}"

        path_to_save = self.__get_patched_folder_and_filename(scene_id, frame_id)

        dir_path = os.path.dirname(path_to_save)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

            OnceFramePatcher.serialise(path=path_to_save,
                                       point_cloud=frame_point_cloud)
        elif self.__overwrite:
            OnceFramePatcher.serialise(path=path_to_save,
                                       point_cloud=frame_point_cloud)
        else:
            print("Skipping frame " + frame_id)

        return path_to_save

    def can_serialise_frame_point_cloud(self,
                                        scene_id: str,
                                        frame_id: str) -> bool:
        assert scene_id in self.__scene_ids, \
            f"Unknown scene id {scene_id}"

        path_to_save = self.__get_patched_folder_and_filename(scene_id, frame_id)

        # We can serialise point cloud if there is no point cloud saved.
        return not os.path.exists(path_to_save)

    def get_frame_point_cloud(self,
                              scene_id: str,
                              frame_id: str) -> np.ndarray:
        assert scene_id in self.__scene_ids, \
            f"Unknown scene id {scene_id}"

        return self.__once.get_frame_point_cloud(scene_id=scene_id,
                                                 frame_id=frame_id)

    def get_instance_point_cloud(self,
                                 scene_id: str,
                                 frame_id: str,
                                 instance_id: str,
                                 frame_point_cloud: np.ndarray) -> np.ndarray:
        assert scene_id in self.__scene_ids, \
            f"Unknown scene id {scene_id}"

        return get_instance_point_cloud(seq_id=scene_id,
                                        frame_id=frame_id,
                                        instance_id=instance_id,
                                        frame_point_cloud=frame_point_cloud,
                                        once=self.__once)

    def __get_patched_folder_and_filename(self, scene_id: str, frame_id: str):
        folder_name = f"{scene_id}_patched"
        patched_filename = f"{frame_id}.bin"

        return os.path.join(self.__dataset_root, 'patched', 'data', folder_name, 'lidar_roof', patched_filename)
