from nuscenes import NuScenes

from src.accumulation.default_accumulator_strategy import DefaultAccumulatorStrategy
from src.accumulation.point_cloud_accumulator import PointCloudAccumulator
from src.utils.nuscenes_helper import group_instances_across_frames
from src.utils.visualisation_helper import visualise_points_cloud


def main():
    nuscenes = NuScenes(version='v1.0-mini', dataroot='./temp/nuscenes', verbose=True)

    grouped_instances = group_instances_across_frames(scene_id=0, nuscenes=nuscenes)

    point_cloud_accumulator = PointCloudAccumulator(step=1,
                                                    scene_id=0,
                                                    grouped_instances=grouped_instances,
                                                    nuscenes=nuscenes)
    default_accumulation_strategy = DefaultAccumulatorStrategy()

    accumulated_point_cloud = point_cloud_accumulator.merge(instance_id='e91afa15647c4c4994f19aeb302c7179',
                                                            accumulation_strategy=default_accumulation_strategy)

    visualise_points_cloud(accumulated_point_cloud.T)


if __name__ == '__main__':
    main()
