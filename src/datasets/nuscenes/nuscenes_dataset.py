from __future__ import annotations

import os
import numpy as np

from nuscenes import NuScenes
from typing import Optional

from src.datasets.dataset import Dataset
from src.datasets.frame_patcher import FramePatcher
from src.datasets.nuscenes.nuscenes_scene_iterator import NuScenesSceneIterator
from src.datasets.nuscenes.nuscenes_frame_patcher import NuscenesFramePatcher
from src.datasets.nuscenes.nuscenes_utils import get_frame_point_cloud, get_instance_point_cloud


class NuscenesDataset(Dataset):
    def __init__(self,
                 version='v1.0-mini',
                 dataroot='./temp/nuscenes'):
        self.__nuscenes = NuScenes(version=version, dataroot=dataroot, verbose=True)

        self.__scenes = self.__nuscenes.scene
        self.__scenes_lookup = {str(i): scene for i, scene in enumerate(self.__scenes)}

    @property
    def dataroot(self) -> str:
        return self.__nuscenes.dataroot

    @property
    def scenes(self) -> list:
        return list(self.__scenes_lookup.keys())

    def get_scene_iterator(self, scene_id: str) -> Dataset.SceneIterator:
        assert scene_id in self.__scenes_lookup

        return NuScenesSceneIterator(scene=self.__scenes_lookup[scene_id],
                                     nuscenes=self.__nuscenes)

    def load_frame_patcher(self,
                           scene_id: str,
                           frame_id: str) -> FramePatcher:
        return NuscenesFramePatcher.load(frame_id=frame_id,
                                         nuscenes=self.__nuscenes)

    def can_serialise_frame_point_cloud(self,
                                        scene_id: str,
                                        frame_id: str) -> bool:
        path_to_save = self.__get_lidarseg_patched_folder_and_filename(frame_id)

        # We can serialise point cloud if there is no point cloud saved.
        return not os.path.exists(path_to_save)

    def serialise_frame_point_clouds(self,
                                     scene_id: str,
                                     frame_id: str,
                                     frame_point_cloud: np.ndarray) -> Optional[str]:
        path_to_save = self.__get_lidarseg_patched_folder_and_filename(frame_id)

        dir_path = os.path.dirname(path_to_save)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        NuscenesFramePatcher.serialise(path=path_to_save,
                                       point_cloud=frame_point_cloud)

        return path_to_save

    def get_frame_point_cloud(self,
                              scene_id: str,
                              frame_id: str) -> np.ndarray:
        return get_frame_point_cloud(frame_id=frame_id,
                                     nuscenes=self.__nuscenes)

    def get_instance_point_cloud(self,
                                 scene_id: str,
                                 frame_id: str,
                                 instance_id: str,
                                 frame_point_cloud: np.ndarray) -> np.ndarray:
        return get_instance_point_cloud(frame_id=frame_id,
                                        frame_point_cloud=frame_point_cloud,
                                        instance_id=instance_id,
                                        nuscenes=self.__nuscenes)

    def __get_lidarseg_patched_folder_and_filename(self, frame_id: str):
        frame = self.__nuscenes.get('sample', frame_id)
        lidarseg_token = frame['data']['LIDAR_TOP']
        lidarseg = self.__nuscenes.get('sample_data', lidarseg_token)

        full_path = os.path.join(self.__nuscenes.dataroot, lidarseg['filename'])

        if not full_path.endswith('.bin'):
            raise Exception(f"File path is not expected {full_path}")

        filename = os.path.basename(full_path)
        # Doing dirname twice we remove LIDAR_TOP folder from the path.
        dirname = os.path.dirname(os.path.dirname(full_path))

        return os.path.join(dirname, 'LIDAR_TOP_PATCHED', filename)
