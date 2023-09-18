from __future__ import annotations

import numpy as np

from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box

from src.patching.frame_patcher import FramePatcher
from src.utils.nuscenes_helper import load_frame_point_cloud, reapply_scene_transformation


class NuscenesFramePatcher(FramePatcher):
    """Patches NuScenes frames with new point clouds.
    """

    def __init__(self,
                 frame_id: str,
                 frame_point_cloud: np.ndarray,
                 nuscenes: NuScenes):
        self.__frame_id = frame_id
        self.__frame_point_cloud = frame_point_cloud
        self.__nuscenes = nuscenes

    @classmethod
    def load(cls,
             frame_id: str,
             nuscenes: NuScenes) -> NuscenesFramePatcher:
        """Creates NuscenesFramePatcher instance.

        :param frame_id: str
            ID of a frame.
        :param nuscenes: 'NuScenes'
            Default NuScenes library facade.
        :return: 'NuscenesFramePatcher'
            A constructed instance.
        """
        lidar_point_cloud = load_frame_point_cloud(frame_id=frame_id,
                                                   nuscenes=nuscenes)
        return NuscenesFramePatcher(frame_id=frame_id,
                                    frame_point_cloud=lidar_point_cloud.points,
                                    nuscenes=nuscenes)

    @classmethod
    def serialise(cls,
                  path: str,
                  point_cloud: np.ndarray):
        """Serialises the given frame into a .bin file.
        """

        if not path.endswith('.bin'):
            raise Exception(f"Supports only bin files, got: {path}")

        points_count = point_cloud.shape[1]

        # NuScenes reads file of mx5 dimension and ignores the last dimension
        # as cls.nbr_dims is 4.
        # scan = np.fromfile(file_name, dtype=np.float32)
        # points = scan.reshape((-1, 5))[:, :cls.nbr_dims()]
        np.append(point_cloud, np.zeros(shape=(1, points_count)), axis=0).T.astype(dtype=np.float32).tofile(path)

    @property
    def frame_id(self) -> str:
        return self.__frame_id

    @property
    def frame(self) -> np.ndarray:
        return self.__frame_point_cloud

    def patch_instance(self,
                       instance_id: str,
                       point_cloud: np.ndarray):
        frame = self.__nuscenes.get('sample', self.__frame_id)

        lidarseg_token = frame['data']['LIDAR_TOP']

        frame_annotations_lookup: set[str] = set(frame['anns'])

        instance_annotations_lookup: set[str] = \
            set(self.__nuscenes.field2token('sample_annotation', 'instance_token', instance_id))

        intersection = list(set.intersection(frame_annotations_lookup, instance_annotations_lookup))

        assert len(intersection) == 1, \
            f"Frame {self.__frame_id} should have the only instance of {instance_id}"

        annotation_token = intersection[0]

        _, boxes, _ = self.__nuscenes.get_sample_data(lidarseg_token, selected_anntokens=[annotation_token])

        assert len(boxes) == 1
        box = boxes[0]

        points = self.__frame_point_cloud[0:3, :]
        mask = points_in_box(box, points)

        # Remove masked elements in frame.
        self.__frame_point_cloud = self.__frame_point_cloud[:, np.where(~mask)[0]]

        # Put the object back into the scene.
        point_cloud = reapply_scene_transformation(annotation_token=annotation_token,
                                                   lidarseg_token=lidarseg_token,
                                                   point_cloud=point_cloud,
                                                   nuscenes=self.__nuscenes)

        # Append instance patch: append should happen along
        self.__frame_point_cloud = np.concatenate((self.__frame_point_cloud, point_cloud), axis=1)
