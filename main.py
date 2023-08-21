from nuscenes import NuScenes

from src.accumulation.default_accumulator_strategy import DefaultAccumulatorStrategy
from src.accumulation.point_cloud_accumulator import PointCloudAccumulator
from src.patching.nuscenes_frame_patcher import NuscenesFramePatcher
from src.utils.nuscenes_helper import group_instances_across_frames
from src.utils.visualisation_helper import visualise_points_cloud


def main():
    nuscenes = NuScenes(version='v1.0-mini', dataroot='./temp/nuscenes', verbose=True)

    grouped_instances = group_instances_across_frames(scene_id=0, nuscenes=nuscenes)

    instances_per_frames_lookup: dict[str, set[str]] = dict()
    for instance, frames in grouped_instances.items():
        for frame in frames:
            if frame not in instances_per_frames_lookup:
                instances_per_frames_lookup[frame] = set()
            instances_per_frames_lookup[frame].add(instance)

    print({i: len(v) for i, v in instances_per_frames_lookup.items()})

    point_cloud_accumulator = PointCloudAccumulator(step=1,
                                                    scene_id=0,
                                                    grouped_instances=grouped_instances,
                                                    nuscenes=nuscenes)
    default_accumulation_strategy = DefaultAccumulatorStrategy()

    frame_id = '9813c23a5f1448b09bb7910fea9baf20'
    instance_ids: set[str] = set()

    for instance_id, frames in grouped_instances.items():
        for frame in frames:
            if frame == frame_id:
                instance_ids.add(instance_id)

    print('Detected', len(instance_ids), 'objects in the frame')

    frame_patcher = NuscenesFramePatcher.load(frame_id=frame_id,
                                              nuscenes=nuscenes)
    # Original unmodified frame.
    visualise_points_cloud(frame_patcher.frame.T)

    for instance_id in instance_ids:
        accumulated_point_cloud = point_cloud_accumulator.merge(instance_id=instance_id,
                                                                accumulation_strategy=default_accumulation_strategy)

        print('frames for', instance_id, 'are', grouped_instances[instance_id])

        # Visualise accumulated point cloud.
        visualise_points_cloud(accumulated_point_cloud.T)

        frame_patcher.patch_instance(instance_id=instance_id,
                                     point_cloud=accumulated_point_cloud)

    # Patched scene.
    visualise_points_cloud(frame_patcher.frame.T)

    scene_file = f"{frame_patcher.frame_id}.bin"
    NuscenesFramePatcher.serialise(path=scene_file,
                                   point_cloud=frame_patcher.frame)
    print('File saved to:', scene_file)


if __name__ == '__main__':
    main()
