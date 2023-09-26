import os
import numpy as np

from src.utils.file_utils import list_all_files_with_extension
from src.utils.geometry_utils import points_in_box


def find_all_scenes(dataset_root: str) -> list:
    """Returns list of scene ids.
    """
    metadata_files = list_all_files_with_extension(files=[dataset_root], extension='pkl')
    return [file[:-4] for file in metadata_files]


def load_frame_descriptors(dataset_root: str,
                           scene_id: str) -> list:
    metadata_file = os.path.join(dataset_root, f"{scene_id}.pkl")

    assert os.path.exists(metadata_file), \
        f"Cannot find file {metadata_file}"

    return np.load(metadata_file, allow_pickle=True)


def count_frames_in_scene(dataset_root: str,
                          scene_id: str) -> int:
    frame_pcr_dir = os.path.join(dataset_root, scene_id)
    pcr_list = list_all_files_with_extension(files=[frame_pcr_dir], extension='.npy', shallow=True)
    return len(pcr_list)


def load_frame_point_cloud(dataset_root: str,
                           scene_id: str,
                           frame_id: str) -> np.ndarray:
    frame_pcr_file = os.path.join(dataset_root, scene_id, f"{frame_id}.npy")

    assert os.path.exists(frame_pcr_file), \
        f"Cannot find file {frame_pcr_file}"

    return np.load(frame_pcr_file)


def get_instance_point_cloud(frame_point_cloud: np.ndarray,
                             instance_id: str,
                             frame_descriptor: dict) -> np.ndarray:
    """Returns point cloud for the given instance in the given frame.

    The returned point cloud has reset rotation and translation.

    :param frame_id: str
        ID of a frame (aka sample).
    :param instance_id: str
        ID of an instance.
    :param nuscenes: 'NuScenes'
        NuScenes dataset facade.
    :param frame_point_cloud: 'LidarPointCloud'
        Point Cloud from lidar.
    :return: np.ndarray[float]
        Returns point cloud for the given object.
        Dimension of the array is 5xm.
    """
    annotations = frame_descriptor['annotations']
    ids = annotations['ids']

    instance_index = np.where(ids == instance_id)
    instance_column = instance_index[0][0]

    center_xyz = annotations['locations'][instance_column, :]
    dimensions_lwh = annotations['dimensions'][instance_column, :]
    heading_angle = annotations['heading_angles'][instance_column]

    print(frame_point_cloud.shape)

    points = frame_point_cloud[:, 0:3].T

    mask = points_in_box(center_xyz=center_xyz,
                         dimensions_lwh=dimensions_lwh,
                         heading_angle=heading_angle,
                         points=points)

    instance_point_cloud = frame_point_cloud[np.where(mask)[0], :]

    return instance_point_cloud
