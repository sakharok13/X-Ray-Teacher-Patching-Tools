from __future__ import annotations

import numpy as np

from src.datasets.frame_patcher import FramePatcher
from src.datasets.waymo.waymo_utils import get_frame_point_cloud, reapply_frame_transformation
from src.utils.geometry_utils import points_in_box


class WaymoFramePatcher(FramePatcher):
    """Patches NuScenes frames with new point clouds.
    """

    def __init__(self,
                 scene_id: str,
                 frame_id: str,
                 frame_point_cloud: np.ndarray,
                 frame_descriptor: dict):
        self.__scene_id = scene_id
        self.__frame_id = frame_id
        self.__frame_point_cloud = frame_point_cloud
        self.__frame_descriptor = frame_descriptor

    @classmethod
    def load(cls,
             dataset_root: str,
             scene_id: str,
             frame_id: str,
             scene_descriptor: dict) -> WaymoFramePatcher:
        lidar_point_cloud = get_frame_point_cloud(dataset_root=dataset_root,
                                                  scene_id=scene_id,
                                                  frame_descriptor=scene_descriptor[frame_id])

        return WaymoFramePatcher(scene_id=scene_id,
                                 frame_id=frame_id,
                                 frame_point_cloud=lidar_point_cloud,
                                 frame_descriptor=scene_descriptor[frame_id])

    @classmethod
    def serialise(cls,
                  path: str,
                  point_cloud: np.ndarray):
        """Serialises the given frame into a .npy file.
        """

        if not path.endswith('.npy'):
            raise Exception(f"Supports only npy files, got: {path}")

        np.save(point_cloud)

    @property
    def scene_id(self) -> str:
        return self.__scene_id

    @property
    def frame_id(self) -> str:
        return self.__frame_id

    @property
    def frame(self) -> np.ndarray:
        return self.__frame_point_cloud

    def patch_instance(self,
                       instance_id: str,
                       point_cloud: np.ndarray):
        annotations = self.__frame_descriptor['annotations']
        ids = annotations['ids']

        instance_index = np.where(ids == instance_id)
        instance_column = instance_index[0][0]

        center_xyz = annotations['locations'][instance_column, :]
        dimensions_lwh = annotations['dimensions'][instance_column, :]
        heading_angle = annotations['heading_angles'][instance_column]

        points = self.__frame_point_cloud[0:3, :]
        mask = points_in_box(center_xyz=center_xyz,
                             dimensions_lwh=dimensions_lwh,
                             heading_angle=heading_angle,
                             points=points)

        # Remove masked elements in frame.
        self.__frame_point_cloud = self.__frame_point_cloud[:, np.where(~mask)[0]]

        # Put the object back into the scene.
        point_cloud = reapply_frame_transformation(point_cloud=point_cloud,
                                                   instance_id=instance_id,
                                                   frame_descriptor=self.__frame_descriptor)

        # Append instance patch: append should happen along
        self.__frame_point_cloud = np.concatenate((self.__frame_point_cloud, point_cloud), axis=1)
