import os
import numpy as np

from pyquaternion import Quaternion

from src.utils.file_utils import list_all_files_with_extension
from src.utils.geometry_utils import points_in_box, transform_matrix


def find_all_scenes(dataset_root: str) -> list:
    """Returns list of scene ids.
    """
    metadata_files = list_all_files_with_extension(files=[dataset_root], extension='pkl')
    return [os.path.basename(file).split('.')[0] for file in metadata_files]


def load_scene_descriptor(dataset_root: str,
                          scene_id: str) -> dict:
    metadata_file = os.path.join(dataset_root, f"{scene_id}.pkl")

    assert os.path.exists(metadata_file), \
        f"Cannot find file {metadata_file}"

    raw_scene_descriptor = np.load(metadata_file, allow_pickle=True)
    return {frame_descriptor['frame_id']: frame_descriptor for frame_descriptor in raw_scene_descriptor}


def count_frames_in_scene(dataset_root: str,
                          scene_id: str) -> int:
    frame_pcr_dir = os.path.join(dataset_root, scene_id)
    pcr_list = list_all_files_with_extension(files=[frame_pcr_dir], extension='.npy', shallow=True)
    return len(pcr_list)


def get_frame_point_cloud(dataset_root: str,
                          scene_id: str,
                          frame_descriptor: dict) -> np.ndarray:
    frame_index = get_frame_index(frame_descriptor)
    frame_pcr_file = os.path.join(dataset_root, scene_id, f"{frame_index:04d}.npy")

    assert os.path.exists(frame_pcr_file), \
        f"Cannot find file {frame_pcr_file}"

    return np.load(frame_pcr_file).T


def get_instance_point_cloud(frame_point_cloud: np.ndarray,
                             instance_id: str,
                             frame_descriptor: dict) -> np.ndarray:
    """Returns point cloud for the given instance in the given frame.

    The returned point cloud has reset rotation and translation.

    :param frame_point_cloud: np.ndarray
        Frame point cloud in <dimension, N> format.
    :param instance_id: str
        ID of an instance.
    :param frame_descriptor: dict
        Descriptor of the given frame.
    :return: np.ndarray[float]
        Returns point cloud for the given object.
        Dimension of the array is 5xm.
    """
    annotations = frame_descriptor['annos']
    ids = annotations['obj_ids']

    # O(obj_ids)
    instance_index = np.where(ids == instance_id)
    instance_column = instance_index[0][0]

    center_xyz = annotations['location'][instance_column, :]
    dimensions_lwh = annotations['dimensions'][instance_column, :]
    heading_angle = annotations['heading_angles'][instance_column]

    points = frame_point_cloud[0:3, :]

    mask = points_in_box(center_xyz=center_xyz,
                         dimensions_lwh=dimensions_lwh,
                         heading_angle=heading_angle,
                         points=points)

    instance_point_cloud = frame_point_cloud[:, np.where(mask)[0]]

    identity_transformation = transform_matrix(center_xyz,
                                               Quaternion(angle=heading_angle, axis=[0, 0, 1]),
                                               inverse=True)

    instance_point_cloud = __apply_transformation_matrix(point_cloud=instance_point_cloud,
                                                         transformation_matrix=identity_transformation)
    return instance_point_cloud


def reapply_frame_transformation(point_cloud: np.ndarray,
                                 instance_id: str,
                                 frame_descriptor: dict) -> np.ndarray:
    annotations = frame_descriptor['annos']
    ids = annotations['obj_ids']

    instance_index = np.where(ids == instance_id)
    instance_column = instance_index[0][0]

    center_xyz = annotations['location'][instance_column, :]
    heading_angle = annotations['heading_angles'][instance_column]

    reverse_transformation = transform_matrix(center_xyz,
                                              Quaternion(angle=heading_angle, axis=[0, 0, 1]),
                                              inverse=False)

    instance_point_cloud = __apply_transformation_matrix(point_cloud=point_cloud,
                                                         transformation_matrix=reverse_transformation)

    return instance_point_cloud


def get_frame_index(frame_descriptor: dict) -> int:
    # Some of converted waymo formats contain frame_index,
    # while others use sample_idx inside of point_cloud
    # to figure out the frame index.
    point_cloud_descriptor = frame_descriptor['point_cloud']
    if 'frame_index' in point_cloud_descriptor:
        frame_index = point_cloud_descriptor['frame_index']
    elif 'sample_idx' in point_cloud_descriptor:
        frame_index = point_cloud_descriptor['sample_idx']
    else:
        raise Exception(f"Frame descriptor does not have frame_index. Descriptor: {frame_descriptor}")

    return frame_index


def __apply_transformation_matrix(point_cloud: np.ndarray,
                                  transformation_matrix: np.ndarray) -> np.ndarray:
    """Applies given transformation matrix to the given point cloud.

    :param point_cloud: np.ndarray[float]
        Point cloud to transform of shape [3, n]
    :param transformation_matrix: np.ndarray[float]
        Transformation matrix that describes rotation and translation of shape [4, 4].
    :return: np.ndarray[float]
        Modified point cloud.
    """
    points_count = point_cloud.shape[1]
    point_cloud[:3, :] = transformation_matrix.dot(
        np.vstack((point_cloud[:3, :], np.ones(points_count))))[:3, :]
    return point_cloud
