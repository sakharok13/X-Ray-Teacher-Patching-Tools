import argparse
import datetime
import os
import numpy as np
import open3d as o3d
import concurrent.futures

from src.accumulation.accumulation_strategy import AccumulationStrategy
from src.accumulation.default_accumulator_strategy import DefaultAccumulatorStrategy
from src.accumulation.gedi_accumulator_strategy import GediAccumulatorStrategy
from src.accumulation.point_cloud_accumulator import PointCloudAccumulator

from src.datasets.dataset import Dataset
from src.datasets.nuscenes.nuscenes_dataset import NuscenesDataset
from src.datasets.waymo.waymo_dataset import WaymoDataset
from src.utils.dataset_helper import group_instances_across_frames
from src.utils.o3d_helper import convert_to_o3d_pointcloud


def __patch_scene(scene_id: str,
                  accumulation_strategy: AccumulationStrategy,
                  dataset: Dataset,
                  export_instances: bool,
                  export_frames: bool) -> bool:
    grouped_instances = group_instances_across_frames(scene_id=scene_id, dataset=dataset)

    point_cloud_accumulator = PointCloudAccumulator(step=1,
                                                    grouped_instances=grouped_instances,
                                                    dataset=dataset)

    instance_accumulated_clouds_lookup = dict()

    current_instance_index = 0
    overall_instances_to_process_count = len(grouped_instances)

    output_folder = './ply_instances_geo/'
    output_folder_frame = './ply_frames_geo/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_folder_frame):
        os.makedirs(output_folder_frame)

    for instance in grouped_instances.keys():
        print(f"Merging {instance}")

        assert instance not in instance_accumulated_clouds_lookup

        accumulated_point_cloud = point_cloud_accumulator.merge(scene_id=scene_id,
                                                                instance_id=instance,
                                                                accumulation_strategy=accumulation_strategy)

        if export_instances:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = os.path.join(output_folder, f"{instance}_{timestamp}.ply")
            accumulated_point_cloud_o3d = convert_to_o3d_pointcloud(
                accumulated_point_cloud.T)  # obj instance accumulated
            o3d.io.write_point_cloud(filename, accumulated_point_cloud_o3d)

        instance_accumulated_clouds_lookup[instance] = accumulated_point_cloud

        current_instance_index += 1
        print(f"Processed {int((current_instance_index / overall_instances_to_process_count) * 100)}%")

    frames_to_instances_lookup: dict = dict()
    for instance, frames in grouped_instances.items():
        for frame_id in frames:
            if frame_id not in frames_to_instances_lookup:
                frames_to_instances_lookup[frame_id] = set()
            frames_to_instances_lookup[frame_id].add(instance)

    current_frame_index = 0
    overall_frames_to_patch_count = len(frames_to_instances_lookup)
    for frame_id, instances in frames_to_instances_lookup.items():
        print(f"Patching {frame_id}")

        patcher = dataset.load_frame_patcher(scene_id=scene_id,
                                             frame_id=frame_id)

        for instance in instances:
            # Make sure you copy instance_accumulated_clouds_lookup[instance]
            # to do not carry the rotation and translation in between frames.
            patcher.patch_instance(instance_id=instance,
                                   point_cloud=np.copy(instance_accumulated_clouds_lookup[instance]))

        saved_path = dataset.serialise_frame_point_clouds(scene_id=scene_id,
                                                          frame_id=frame_id,
                                                          frame_point_cloud=patcher.frame)

        if export_frames:
            filename = os.path.join(output_folder_frame, f"{current_frame_index}.ply")
            frame_o3d = convert_to_o3d_pointcloud(patcher.frame.T)  # patched frame
            o3d.io.write_point_cloud(filename, frame_o3d)

        current_frame_index += 1

        if saved_path is not None:
            print(f"{int((current_frame_index / overall_frames_to_patch_count) * 100)}%, saved to {saved_path}")
        else:
            print(f"There was an error saving the point cloud")

    # Return OK status when finished processing.
    return True


def __process_batch(scenes: list,
                    dataset: Dataset,
                    accumulation_strategy: AccumulationStrategy,
                    export_instances: bool,
                    export_frames: bool):
    for scene_id in scenes:
        __patch_scene(scene_id=str(scene_id),
                      dataset=dataset,
                      accumulation_strategy=accumulation_strategy,
                      export_instances=export_instances,
                      export_frames=export_frames)


def __process_dataset(dataset: Dataset,
                      accumulation_strategy: AccumulationStrategy,
                      export_instances: bool,
                      export_frames: bool,
                      num_workers: int):
    assert f"num_workers should be positive {num_workers}", \
        num_workers > 0

    scenes = dataset.scenes
    scenes_count = len(scenes)

    batches = list()
    batch_size = scenes_count // num_workers

    for batch_index in range(num_workers):
        batch_start = batch_index * batch_size
        batch_finish = (batch_index + 1) * batch_size

        if batch_index + 1 == num_workers and scenes_count % num_workers > 0:
            # Last batch will be a bit bigger if there is not enough elements
            # at the end to end up in a standalone batch.
            batch_finish = len(scenes)

        batches.append(scenes[batch_start:batch_finish])

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(__process_batch,
                     batches,
                     [dataset] * num_workers,
                     [accumulation_strategy] * num_workers,
                     [export_instances] * num_workers,
                     [export_frames] * num_workers)


accumulator_strategies = {
    'default': DefaultAccumulatorStrategy(),
    'gedi': GediAccumulatorStrategy(),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="patch scene arguments")
    parser.add_argument("--dataset", type=str, choices=['nuscenes', 'waymo'], default='nuscenes', help="Dataset.")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="NuScenes version.")
    parser.add_argument("--dataroot", type=str, default="./temp/nuscenes", help="Data root location.")
    parser.add_argument("--strategy", type=str, default="default", help="Accumulation strategy.")
    parser.add_argument("--instances", action="store_true", help="Export instances.")
    parser.add_argument("--frames", action="store_true", help="Export frames.")
    parser.add_argument('--start_scene_index', type=int, default=0, help='Specify your scene index to start with.')
    parser.add_argument('--num_workers', type=int, default=1, choices=range(1, 128), help='Count of parallel workers.')
    return parser.parse_args()


def main():
    args = parse_arguments()

    dataset_type = args.dataset

    if dataset_type == 'nuscenes':
        dataset = NuscenesDataset(version=args.version, dataroot=args.dataroot)
    elif dataset_type == 'waymo':
        dataset = WaymoDataset(dataset_root=args.dataroot)
    else:
        raise Exception(f"Unknown dataset {dataset_type}")

    accumulator_strategy = accumulator_strategies[args.strategy]

    __process_dataset(dataset=dataset,
                      accumulation_strategy=accumulator_strategy,
                      export_instances=args.instances,
                      export_frames=args.frames,
                      num_workers=args.num_workers)


if __name__ == '__main__':
    main()
