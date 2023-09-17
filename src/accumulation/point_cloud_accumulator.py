import numpy as np

from src.accumulation.accumulation_strategy import AccumulationStrategy
from src.datasets.dataset import Dataset


class PointCloudAccumulator:
    """Accumulates point cloud of the instances across the entire scene.
    """

    def __init__(self,
                 step: int,
                 grouped_instances: dict,
                 dataset: Dataset):
        assert step > 0, \
            f"Step should be greater than 0, but got {step}"

        self.__step = step
        self.__grouped_instances = grouped_instances
        self.__dataset = dataset

    def merge(self,
              instance_id: str,
              accumulation_strategy: AccumulationStrategy) -> np.ndarray:
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

        instance_frames: list = self.__grouped_instances[instance_id]

        assert len(instance_frames) > 0, \
            f"Instance has not been detected in any frames"

        first_frame_id = instance_frames[0]
        first_frame_point_cloud = self.__dataset.get_frame_point_cloud(frame_id=first_frame_id)

        current_point_cloud = self.__dataset.get_instance_point_cloud(frame_id=first_frame_id,
                                                                      instance_id=instance_id,
                                                                      frame_point_cloud=first_frame_point_cloud)

        for i in range(self.__step, len(instance_frames), self.__step):
            frame_id = instance_frames[i]

            frame_point_cloud = self.__dataset.get_frame_point_cloud(frame_id=frame_id)

            next_point_cloud = self.__dataset.get_instance_point_cloud(frame_id=frame_id,
                                                                       instance_id=instance_id,
                                                                       frame_point_cloud=frame_point_cloud)

            current_point_cloud = accumulation_strategy.on_merge(initial_point_cloud=current_point_cloud,
                                                                 next_point_cloud=next_point_cloud,
                                                                 frame_no=i)

        return current_point_cloud
