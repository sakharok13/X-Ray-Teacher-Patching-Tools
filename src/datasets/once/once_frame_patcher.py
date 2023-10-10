from __future__ import annotations

import numpy as np

from src.datasets.once.once_utils import ONCE
from src.datasets.frame_patcher import FramePatcher
from src.datasets.once.once_utils import get_frame_point_cloud, reapply_frame_transformation
from src.utils.geometry_utils import points_in_box


class OnceFramePatcher(FramePatcher):
    """Patches Once frames with new point clouds.
    """

    def __init__(self,
                 seq_id: str,
                 frame_id: str,
                 frame_point_cloud: np.ndarray,
                 frame_descriptor: dict,
                 once: ONCE):
        self.__seq_id = seq_id
        self.__frame_id = frame_id
        self.__frame_point_cloud = frame_point_cloud
        self.__frame_descriptor = frame_descriptor
        self.__once = once

    @classmethod
    def load(cls,
             seq_id: str,
             frame_id: str,
             once: ONCE) -> OnceFramePatcher:
        """Creates OnceFramePatcher instance.
        :param seq_id: str
            ID of a scene.
        :param frame_id: str
            ID of a frame.
        :param once: 'ONCE'
            ONCE dataset class.
        :return: 'OnceFramePatcher'
            A constructed instance.
        """
        lidar_point_cloud = get_frame_point_cloud(seq_id=seq_id,
                                                  frame_id=frame_id,
                                                  once=once)
        return OnceFramePatcher(seq_id=seq_id,
                                frame_id=frame_id,
                                frame_point_cloud=lidar_point_cloud,
                                once=once)

    @classmethod
    def serialise(cls,
                  path: str,
                  point_cloud: np.ndarray):
        """Serialises the given frame into a .bin file.
        """

        if not path.endswith('.bin'):
            raise Exception(f"Supports only bin files, got: {path}")

        points_count = point_cloud.shape[1]

        np.append(
            point_cloud,
            np.zeros(
                shape=(
                    1,
                    points_count)),
            axis=0).T.astype(
                dtype=np.float32).tofile(path)

    @property
    def frame_id(self) -> str:
        return self.__frame_id

    @property
    def frame(self) -> np.ndarray:
        return self.__frame_point_cloud

    def patch_instance(self,
                       instance_id: str,
                       point_cloud: np.ndarray):
        annotations = self.__once.get_frame_anno(
            self.__seq_id, self.__frame_id)
        ids = annotations['instance_ids']

        instance_index = np.where(ids == instance_id)
        # box - cx, cy, cz, l, w, h, θ
        box = annotations['boxes_3d'][instance_index]
        center_xyz = box[0:3]
        dimensions_lwh = np.array([box[3], box[4], box[5]])
        heading_angle = box[6]

        points = self.__frame_point_cloud[0:3, :]
        mask = points_in_box(center_xyz=center_xyz,
                             dimensions_lwh=dimensions_lwh,
                             heading_angle=heading_angle,
                             points=points)

        # Remove masked elements in frame.
        self.__frame_point_cloud = self.__frame_point_cloud[:, np.where(~mask)[
            0]]

        # Put the object back into the scene.
        point_cloud = reapply_frame_transformation(point_cloud=point_cloud,
                                                   instance_id=instance_id,
                                                   frame_descriptor=self.__frame_descriptor)

        # Append instance patch: append should happen along
        self.__frame_point_cloud = np.concatenate(
            (self.__frame_point_cloud, point_cloud), axis=1)
