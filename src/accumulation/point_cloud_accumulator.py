import numpy as np

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from src.accumulation.accumulation_strategy import AccumulationStrategy
from src.utils.nuscenes_helper import get_instance_point_cloud, load_frame_point_cloud


class PointCloudAccumulator:
    """Accumulates point cloud of the instances across the entire scene.
    """

    def __init__(self,
                 step: int,
                 scene_id: int,
                 grouped_instances: dict[str, list[str]],
                 nuscenes: NuScenes):
        assert step > 0, \
            f"Step should be greater than 0, but got {step}"

        self.__step = step
        self.__scene_id = scene_id
        self.__grouped_instances = grouped_instances
        self.__nuscenes = nuscenes

    def merge(self,
              instance_id: str,
              accumulation_strategy: AccumulationStrategy) -> np.ndarray[float]:
        """Accumulates the point cloud of the given object across the scene using accumulation strategy.

        :param instance_id: str
            ID of an instance.
        :param accumulation_strategy: 'AccumulationStrategy'
            A strategy to concatenate 2 point clouds.
        :return: np.ndarray[float]
            A point cloud accumulated across the entire scene.
        """

        assert instance_id in self.__grouped_instances, \
            f"Unknown instance_id {instance_id}"

        instance_frames: list[str] = self.__grouped_instances[instance_id]

        assert len(instance_frames) > 0, \
            f"Instance has not been detected in any frames"

        first_frame_id = instance_frames[0]
        first_frame_point_cloud = load_frame_point_cloud(frame_id=first_frame_id,
                                                         nuscenes=self.__nuscenes)

        current_point_cloud = self.__load_instance_point_cloud(frame_id=first_frame_id,
                                                               instance_id=instance_id,
                                                               frame_point_cloud=first_frame_point_cloud)

        for i in range(self.__step, len(instance_frames), self.__step):
            frame_id = instance_frames[i]

            frame_point_cloud = load_frame_point_cloud(frame_id=frame_id,
                                                       nuscenes=self.__nuscenes)

            next_point_cloud = self.__load_instance_point_cloud(frame_id=frame_id,
                                                                instance_id=instance_id,
                                                                frame_point_cloud=frame_point_cloud)

            current_point_cloud = accumulation_strategy.on_merge(initial_point_cloud=current_point_cloud,
                                                                 next_point_cloud=next_point_cloud)

        return current_point_cloud

    def __load_instance_point_cloud(self,
                                    frame_id: str,
                                    instance_id: str,
                                    frame_point_cloud: LidarPointCloud) -> np.ndarray[float]:
        return get_instance_point_cloud(frame_id=frame_id,
                                        instance_id=instance_id,
                                        frame_point_cloud=frame_point_cloud,
                                        nuscenes=self.__nuscenes)
