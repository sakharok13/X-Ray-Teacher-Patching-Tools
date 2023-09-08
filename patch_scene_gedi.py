import os.path
import datetime

import numpy as np
from nuscenes import NuScenes
import open3d as o3d

from src.accumulation.accumulation_strategy import AccumulationStrategy
from src.accumulation.gedi_accumulator_strategy import GediAccumulatorStrategy
from src.accumulation.point_cloud_accumulator import PointCloudAccumulator
from src.patching.nuscenes_frame_patcher import NuscenesFramePatcher
from src.utils.nuscenes_helper import group_instances_across_frames
from src.utils.o3d_helper import convert_to_o3d_pointcloud


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
                  accumulation_strategy: GediAccumulatorStrategy,
                  nuscenes: NuScenes):
    grouped_instances = group_instances_across_frames(
        scene_id=scene_id, nuscenes=nuscenes)

    point_cloud_accumulator = PointCloudAccumulator(
        step=1,
        scene_id=scene_id,
        grouped_instances=grouped_instances,
        nuscenes=nuscenes)

    instance_accumulated_clouds_lookup: dict[str, np.ndarray[float]] = dict()

    current_instance_index = 0
    overall_instances_to_process_count = len(grouped_instances)

    output_folder = './ply_instances/'
    output_folder_frame = './ply_frames/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_folder_frame):
        os.makedirs(output_folder_frame)

    for instance in grouped_instances.keys():
        print(f"Merging {instance}")

        assert instance not in instance_accumulated_clouds_lookup

        accumulated_point_cloud = point_cloud_accumulator.merge(
            instance_id=instance, accumulation_strategy=accumulation_strategy)

        # uncomment to save accumulated instances as .ply files
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join(output_folder, f"{instance}_{timestamp}.ply")
        accumulated_point_cloud_o3d = convert_to_o3d_pointcloud(accumulated_point_cloud.T)  # obj instance accumulated
        o3d.io.write_point_cloud(filename, accumulated_point_cloud_o3d)
        """

        instance_accumulated_clouds_lookup[instance] = accumulated_point_cloud

        current_instance_index += 1
        print(
            f"Processed {int((current_instance_index / overall_instances_to_process_count) * 100)}%")

    frames_to_instances_lookup: dict[str, set[str]] = dict()
    for instance, frames in grouped_instances.items():
        for frame_id in frames:
            if frame_id not in frames_to_instances_lookup:
                frames_to_instances_lookup[frame_id] = set()
            frames_to_instances_lookup[frame_id].add(instance)

    current_frame_index = 0
    overall_frames_to_patch_count = len(frames_to_instances_lookup)
    for frame_id, instances in frames_to_instances_lookup.items():
        print(f"Patching {frame_id}")

        patcher = NuscenesFramePatcher.load(frame_id=frame_id,
                                            nuscenes=nuscenes)

        for instance in instances:
            # Make sure you copy instance_accumulated_clouds_lookup[instance]
            # to do not carry the rotation and translation in between frames.
            patcher.patch_instance(
                instance_id=instance, point_cloud=np.copy(
                    instance_accumulated_clouds_lookup[instance]))

        path_to_save = __get_lidarseg_patched_folder_and_filename(
            frame_id=frame_id, nuscenes=nuscenes)

        dir_path = os.path.dirname(path_to_save)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        NuscenesFramePatcher.serialise(path=path_to_save,
                                       point_cloud=patcher.frame)

        # uncomment to save patched frames as .ply files
        """
        filename = os.path.join(output_folder_frame, f"{current_frame_index}.ply")
        frame_o3d = convert_to_o3d_pointcloud(patcher.frame.T)  # patched frame
        o3d.io.write_point_cloud(filename, frame_o3d)
        """


        current_frame_index += 1
        print(f"{int((current_frame_index / overall_frames_to_patch_count) * 100)}%, saved to {path_to_save}")


def main():
    nuscenes = NuScenes(
        version='v1.0-mini',
        dataroot='./data/sets/nuscenes',
        verbose=True)

    scenes = nuscenes.scene
    for scene_id in range(len(scenes)):
        __patch_scene(scene_id=scene_id,
                      accumulation_strategy=GediAccumulatorStrategy(),
                      nuscenes=nuscenes)


if __name__ == '__main__':
    main()
