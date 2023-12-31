from __future__ import annotations

import numpy as np

from src.datasets.once.once_utils import ONCE
from src.datasets.frame_patcher import FramePatcher
from src.datasets.once.once_utils import reapply_frame_transformation, get_frame_instance_ids, get_pickle_data, build_frame_id_to_annotations_lookup
from src.utils.geometry_utils import points_in_box


class OnceFramePatcher(FramePatcher):
    """Patches Once frames with new point clouds.
    """

    def __init__(self,
                 sсene_id: str,
                 frame_id: str,
                 frame_point_cloud: np.ndarray,
                 once: ONCE):
        self.__scene_id = sсene_id
        self.__frame_id = frame_id
        self.__frame_point_cloud = frame_point_cloud
        self.__once = once

        self.__pickle_data = get_pickle_data(self.__once.dataset_root, self.__scene_id)
        self.__frame_id_to_annotations_lookup = build_frame_id_to_annotations_lookup(self.__pickle_data)

    @classmethod
    def load(cls,
             scene_id: str,
             frame_id: str,
             once: ONCE) -> OnceFramePatcher:
        """Creates OnceFramePatcher instance.
        :param scene_id: str
            ID of a scene.
        :param frame_id: str
            ID of a frame.
        :param once: 'ONCE'
            ONCE dataset class.
        :return: 'OnceFramePatcher'
            A constructed instance.
        """
        frame_point_cloud = once.get_frame_point_cloud(scene_id=scene_id,
                                                       frame_id=frame_id)
        return OnceFramePatcher(sсene_id=scene_id,
                                frame_id=frame_id,
                                frame_point_cloud=frame_point_cloud,
                                once=once)

    @classmethod
    def serialise(cls,
                  path: str,
                  point_cloud: np.ndarray):
        """Serialises the given frame into a .bin file.
        """

        if not path.endswith('.bin'):
            raise Exception(f"Supports only bin files, got: {path}")

        point_cloud.T.astype(dtype=np.float32).tofile(path)

    @property
    def frame_id(self) -> str:
        return self.__frame_id

    @property
    def frame(self) -> np.ndarray:
        return self.__frame_point_cloud

    def patch_instance(self,
                       instance_id: str,
                       point_cloud: np.ndarray):
        frame_descriptor = self.__frame_id_to_annotations_lookup[self.__frame_id]
        annotations = frame_descriptor['annos']

        ids = get_frame_instance_ids(self.__scene_id, self.__frame_id, self.__once)

        instance_index = ids.index(instance_id)
        boxes = annotations['boxes_3d']
        box = boxes[instance_index]
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
                                                   frame_descriptor=frame_descriptor,
                                                   instance_id=instance_id,
                                                   once=self.__once)

        # Append instance patch: append should happen along
        if point_cloud.size != 0:
            self.__frame_point_cloud = np.concatenate(
                (self.__frame_point_cloud, point_cloud), axis=1)
