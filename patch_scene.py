import argparse
import logging
import multiprocessing
import os
import sys
import time
from functools import partial
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager, Pool

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), './gedi'))
from src.accumulation.accumulation_strategy import AccumulationStrategy
from src.accumulation.default_accumulator_strategy import DefaultAccumulatorStrategy
from src.accumulation.gedi_accumulator_strategy import GediAccumulatorStrategy
from src.accumulation.greedy_grid_accumulator_strategy import GreedyGridAccumulatorStrategy
from src.accumulation.point_cloud_accumulator import PointCloudAccumulator

from src.datasets.dataset import Dataset
from src.datasets.nuscenes.nuscenes_dataset import NuscenesDataset
from src.datasets.once.once_dataset import OnceDataset
from src.datasets.waymo.waymo_dataset import WaymoDataset
from src.utils.dataset_helper import group_instances_across_frames, can_skip_frame, can_skip_scene
from src.utils.logging_utils import create_root_handler


def __patch_scene(scene_id: str,
                  accumulation_strategy: AccumulationStrategy,
                  dataset: Dataset,
                  force_overwrite: bool,
                  gedi_counter):
    # O(frames)
    if can_skip_scene(dataset=dataset,
                      scene_id=scene_id,
                      force_overwrite=force_overwrite):
        logging.info(f"[Scene {scene_id}] Skipping scene.")
        return True

    if isinstance(accumulation_strategy, GediAccumulatorStrategy):
        gpu_id = -1
        while gpu_id == -1:
            for i in range(torch.cuda.device_count()):
                if gedi_counter[i] < 4:
                    gpu_id = i
                    gedi_counter[i] += 1
                    break
            if gpu_id == -1:
                time.sleep(1)  # wait for a GPU to be available

        torch.cuda.set_device(gpu_id)
        logging.info(f"Running GediAccumulatorStrategy on GPU {gpu_id}")

    logging.info(f"[Scene {scene_id}] Starting...")

    # O(frames * instances)
    grouped_instances = group_instances_across_frames(scene_id=scene_id, dataset=dataset)

    point_cloud_accumulator = PointCloudAccumulator(step=1,
                                                    grouped_instances=grouped_instances,
                                                    dataset=dataset)

    instance_accumulated_clouds_lookup = dict()

    current_instance_index = 0
    overall_instances_to_process_count = len(grouped_instances)

    # O(instances * frames * N * d)
    for instance in grouped_instances.keys():
        logging.info(f"[Scene {scene_id}] Merging {instance}")

        assert instance not in instance_accumulated_clouds_lookup

        # O(frames * N * d)
        accumulated_point_cloud = point_cloud_accumulator.merge(scene_id=scene_id,
                                                                instance_id=instance,
                                                                accumulation_strategy=accumulation_strategy)

        instance_accumulated_clouds_lookup[instance] = accumulated_point_cloud

        current_instance_index += 1
        logging.info(
            f"[Scene {scene_id}] Merged {int((current_instance_index / overall_instances_to_process_count) * 100)}% "
            f"of instances.")

    frames_to_instances_lookup: dict = dict()
    # O(instances * frames)
    for instance, frames in grouped_instances.items():
        for frame_id in frames:
            if can_skip_frame(dataset=dataset,
                              scene_id=scene_id,
                              frame_id=frame_id,
                              force_overwrite=force_overwrite):
                logging.warning(f"[Scene {scene_id}] Skipping frame {frame_id}...")
                continue

            if frame_id not in frames_to_instances_lookup:
                frames_to_instances_lookup[frame_id] = set()
            frames_to_instances_lookup[frame_id].add(instance)

    current_frame_index = 0
    overall_frames_to_patch_count = len(frames_to_instances_lookup)
    logging.info(f"[Scene {scene_id}] Found {overall_frames_to_patch_count} frames to patch.")

    # O(instances * frames)
    for frame_id, instances in frames_to_instances_lookup.items():
        logging.info(f"[Scene {scene_id}] Patching frame {frame_id}...")

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

        current_frame_index += 1

        if saved_path is not None:
            logging.info(f"[Scene {scene_id}] {int((current_frame_index / overall_frames_to_patch_count) * 100)}%, "
                         f"saved to {saved_path}")
        else:
            logging.error(f"[Scene {scene_id}] There was an error saving the point cloud for frame {frame_id}")

    logging.info(f"[Scene {scene_id}] Wrapping up.")

    # If the strategy is GediAccumulatorStrategy, decrease the counter
    if isinstance(accumulation_strategy, GediAccumulatorStrategy):
        gedi_counter[gpu_id] -= 1

    # Return OK status when finished processing.
    return True


def __on_process_init(log_queue,
                      enable_logging: bool):
    queue_handler = QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.disabled = not enable_logging
    logger.setLevel(logging.INFO)
    logger.addHandler(queue_handler)


def __process_dataset(dataset: Dataset,
                      accumulation_strategy: AccumulationStrategy,
                      num_workers: int,
                      force_overwrite: bool,
                      enable_logging: bool):
    assert num_workers > 0, "num_workers should be positive"

    print(f"Processing dataset from: {dataset.dataroot}")

    scenes = dataset.scenes
    scenes_count = len(scenes)

    with Manager() as manager:
        log_queue = manager.Queue()
        queue_listener = QueueListener(log_queue, create_root_handler())
        queue_listener.start()

        gedi_counter = manager.dict()
        for i in range(torch.cuda.device_count()):
            gedi_counter[i] = 0

        patch_scene = partial(
            __patch_scene,
            accumulation_strategy=accumulation_strategy,
            dataset=dataset,
            force_overwrite=force_overwrite,
            gedi_counter=gedi_counter
        )

        with Pool(num_workers, __on_process_init, [log_queue, enable_logging]) as p:
            list(tqdm(p.imap_unordered(patch_scene, scenes), total=scenes_count))

        # Close the queue and the handler_process.
        queue_listener.stop()


accumulator_strategies = {
    'default': DefaultAccumulatorStrategy(),
    'gedi': GediAccumulatorStrategy(),
    'greedy_grid': GreedyGridAccumulatorStrategy(),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description='patch scene arguments')
    parser.add_argument('--dataset', type=str, choices=['nuscenes', 'once', 'waymo'], default='nuscenes',
                        help='Dataset.')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='NuScenes version.')
    parser.add_argument("--split", type=str, choices=['train', 'test', 'val', 'raw_small', 'raw_medium', 'raw_large'],
                        default="train", help="Once dataset split type.")
    parser.add_argument('--dataroot', type=str, default='./temp/nuscenes', help='Data root location.')
    parser.add_argument('--strategy', type=str, default='default', choices=accumulator_strategies.keys(),
                        help='Accumulation strategy.')
    parser.add_argument('--enable_logging', action='store_true', help='Save additional logs to file.')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(),
                        help='Count of parallel workers.')
    parser.add_argument('--force_overwrite', action='store_true', help='Overwrite saved files.')
    return parser.parse_args()


def main():
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_arguments()

    dataset_type = args.dataset

    if dataset_type == 'nuscenes':
        dataset = NuscenesDataset(version=args.version, dataroot=args.dataroot)
    elif dataset_type == 'once':
        dataset = OnceDataset(split=args.split, dataset_root=args.dataroot)
    elif dataset_type == 'waymo':
        dataset = WaymoDataset(dataset_root=args.dataroot)
    else:
        raise Exception(f"Unknown dataset {dataset_type}")

    accumulator_strategy = accumulator_strategies[args.strategy]

    __process_dataset(dataset=dataset,
                      accumulation_strategy=accumulator_strategy,
                      num_workers=args.num_workers,
                      force_overwrite=args.force_overwrite,
                      enable_logging=args.enable_logging)


if __name__ == '__main__':
    main()
