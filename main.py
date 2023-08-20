from nuscenes import NuScenes

from src.accumulation.default_accumulator_strategy import DefaultAccumulatorStrategy
from src.accumulation.point_cloud_accumulator import PointCloudAccumulator
from src.patching.nuscenes_frame_patcher import NuscenesFramePatcher
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

    instance_id = 'e91afa15647c4c4994f19aeb302c7179'

    accumulated_point_cloud = point_cloud_accumulator.merge(instance_id=instance_id,
                                                            accumulation_strategy=default_accumulation_strategy)

    print('frames for', instance_id, 'are', grouped_instances[instance_id])

    visualise_points_cloud(accumulated_point_cloud.T)

    frame_patcher = NuscenesFramePatcher.load(frame_id='39586f9d59004284a7114a68825e8eec',
                                              nuscenes=nuscenes)

    # Original unmodified frame.
    visualise_points_cloud(frame_patcher.frame.T)

    frame_patcher.patch_instance(instance_id=instance_id,
                                 point_cloud=accumulated_point_cloud)

    # Patched scene.
    visualise_points_cloud(frame_patcher.frame.T)


if __name__ == '__main__':
    main()
