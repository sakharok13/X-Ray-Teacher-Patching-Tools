import argparse
import os.path

import numpy as np
from nuscenes import NuScenes

from src.accumulation.accumulation_strategy import AccumulationStrategy
from src.accumulation.default_accumulator_strategy import DefaultAccumulatorStrategy
from src.accumulation.point_cloud_accumulator import PointCloudAccumulator
from src.patching.nuscenes_frame_patcher import NuscenesFramePatcher
from src.utils.nuscenes_helper import group_instances_across_frames


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--start_scene_index', type=int, default=0, help='specify your scene index to start with')
    args = parser.parse_args()
    
    return args

def __get_lidarseg_patched_folder_and_filename(frame_id: str,
                                               nuscenes: NuScenes):
    frame = nuscenes.get('sample', frame_id)
    lidarseg_token = frame['data']['LIDAR_TOP']
    lidarseg = nuscenes.get('sample_data', lidarseg_token)

    full_path = os.path.join(nuscenes.dataroot, lidarseg['filename'])

    if not full_path.endswith('.bin'):
        raise Exception(f"File path is not expected {full_path}")

    filename = os.path.basename(full_path)
    # Doing dirname twice we remove LIDAR_TOP folder from the path.
    dirname = os.path.dirname(os.path.dirname(full_path))

    return os.path.join(dirname, 'LIDAR_TOP_PATCHED', filename)


def __patch_scene(scene_id: int,
                  accumulation_strategy: AccumulationStrategy,
                  nuscenes: NuScenes):
    grouped_instances = group_instances_across_frames(scene_id=scene_id, nuscenes=nuscenes)

    point_cloud_accumulator = PointCloudAccumulator(step=1,
                                                    scene_id=scene_id,
                                                    grouped_instances=grouped_instances,
                                                    nuscenes=nuscenes)

    instance_accumulated_clouds_lookup: dict = dict()

    current_instance_index = 0
    overall_instances_to_process_count = len(grouped_instances)
    
    check = 0 
    for instance, frames in grouped_instances.items():
        for frame_id in frames:
            path_to_save = __get_lidarseg_patched_folder_and_filename(frame_id=frame_id,
                                                                      nuscenes=nuscenes)
            if os.path.exists(path_to_save):
                print(f'Skipping frame {frame_id}')
                continue
            else:
                check = 1
    if check == 0:
        return 0
                    
    for instance in grouped_instances.keys():
        
        print(f"Merging {instance}")

        assert instance not in instance_accumulated_clouds_lookup

        accumulated_point_cloud = point_cloud_accumulator.merge(instance_id=instance,
                                                                accumulation_strategy=accumulation_strategy)
        instance_accumulated_clouds_lookup[instance] = accumulated_point_cloud

        current_instance_index += 1
        print(f"Processed {int((current_instance_index / overall_instances_to_process_count) * 100)}%")

    frames_to_instances_lookup: dict = dict()
    for instance, frames in grouped_instances.items():
        for frame_id in frames:
            if frame_id not in frames_to_instances_lookup:
                path_to_save = __get_lidarseg_patched_folder_and_filename(frame_id=frame_id,
                                                                          nuscenes=nuscenes)
                if os.path.exists(path_to_save):
                    print(f'Skipping frame {frame_id}')
                    continue
                    
                frames_to_instances_lookup[frame_id] = set()
            frames_to_instances_lookup[frame_id].add(instance)

    current_frame_index = 0
    overall_frames_to_patch_count = len(frames_to_instances_lookup)
    for frame_id, instances in frames_to_instances_lookup.items():
        path_to_save = __get_lidarseg_patched_folder_and_filename(frame_id=frame_id,
                                                          nuscenes=nuscenes)
        if os.path.exists(path_to_save):
            print(f'Skipping frame {frame_id}')
            continue
            
        print(f"Patching {frame_id}")

        patcher = NuscenesFramePatcher.load(frame_id=frame_id,
                                            nuscenes=nuscenes)

        for instance in instances:
            # Make sure you copy instance_accumulated_clouds_lookup[instance]
            # to do not carry the rotation and translation in between frames.
            patcher.patch_instance(instance_id=instance,
                                   point_cloud=np.copy(instance_accumulated_clouds_lookup[instance]))

        dir_path = os.path.dirname(path_to_save)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        NuscenesFramePatcher.serialise(path=path_to_save,
                                       point_cloud=patcher.frame)

        current_frame_index += 1
        print(f"{int((current_frame_index / overall_frames_to_patch_count) * 100)}%, saved to {path_to_save}")


def main():
    args = parse_config()
    
    nuscenes = NuScenes(version='v1.0-trainval', dataroot='../nuscenes/v1.0-trainval', verbose=True)

    scenes = nuscenes.scene
    length = len(scenes)

    for scene_id in range(args.start_scene_index, length):
        __patch_scene(scene_id=scene_id,
                      accumulation_strategy=DefaultAccumulatorStrategy(),
                      nuscenes=nuscenes)
        progress = (scene_id + 1 - args.start_scene_index) / (length - args.start_scene_index) * 100
        print('LOCAL PROGRESS: %.2f' % progress + '%')


if __name__ == '__main__':
    main()
