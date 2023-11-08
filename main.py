from src.accumulation.default_accumulator_strategy import DefaultAccumulatorStrategy
from src.accumulation.point_cloud_accumulator import PointCloudAccumulator

from src.datasets.dataset import Dataset
from src.utils.dataset_helper import group_instances_across_frames
from src.datasets.nuscenes.nuscenes_dataset import NuscenesDataset
from src.datasets.waymo.waymo_dataset import WaymoDataset
from src.datasets.once.once_dataset import OnceDataset

from src.utils.visualisation_helper import visualise_points_cloud


def __create_dataset(dataset: str) -> Dataset:
    if dataset == 'nuscenes':
        return NuscenesDataset(version='v1.0-mini', dataroot='./temp/nuscenes')
    elif dataset == 'waymo':
        return WaymoDataset(dataset_root='./temp/open-waymo')
    elif dataset == 'once':
        return OnceDataset(dataset_root='./temp/once', split="raw_small")
    else:
        raise Exception(f"Unknown dataset {dataset}")


def main():
    dataset = __create_dataset('once')

    print('Dataset root:', dataset.dataroot)
    print('Detected scenes:', dataset.scenes)

    scene_id = '000013'

    grouped_instances = group_instances_across_frames(scene_id=scene_id, dataset=dataset)

    instances_per_frames_lookup = dict()
    for instance, frames in grouped_instances.items():
        for frame in frames:
            if frame not in instances_per_frames_lookup:
                instances_per_frames_lookup[frame] = set()
            instances_per_frames_lookup[frame].add(instance)

    print({i: len(v) for i, v in instances_per_frames_lookup.items()})

    point_cloud_accumulator = PointCloudAccumulator(step=1,
                                                    grouped_instances=grouped_instances,
                                                    dataset=dataset)
    default_accumulation_strategy = DefaultAccumulatorStrategy()

    frame_id = '1616013899200'
    instance_ids = set()

    for instance_id, frames in grouped_instances.items():
        for frame in frames:
            if frame == frame_id:
                instance_ids.add(instance_id)

    print('Detected', len(instance_ids), 'objects in the frame')

    frame_patcher = dataset.load_frame_patcher(scene_id=scene_id, frame_id=frame_id)
    # Original unmodified frame.
    visualise_points_cloud(frame_patcher.frame.T)

    for i, instance_id in enumerate(instance_ids):
        accumulated_point_cloud = point_cloud_accumulator.merge(scene_id=scene_id,
                                                                instance_id=instance_id,
                                                                accumulation_strategy=default_accumulation_strategy)

        print('Frames for instance', instance_id, 'are', grouped_instances[instance_id])

        # Visualise accumulated point cloud.
        # visualise_points_cloud(accumulated_point_cloud.T)

        frame_patcher.patch_instance(instance_id=instance_id,
                                     point_cloud=accumulated_point_cloud)

        print('Progress', f"{(i/len(instance_ids) * 100):.1f}%")

    # Patched scene.
    visualise_points_cloud(frame_patcher.frame.T)

    saved_path = dataset.serialise_frame_point_clouds(scene_id=scene_id,
                                                      frame_id=frame_patcher.frame_id,
                                                      frame_point_cloud=frame_patcher.frame)
    print('File saved to:', saved_path)


if __name__ == '__main__':
    main()
