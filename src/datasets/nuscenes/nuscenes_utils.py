import os
import numpy as np

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix


def get_instance_point_cloud(frame_id: str,
                             frame_point_cloud: np.ndarray,
                             instance_id: str,
                             nuscenes: NuScenes) -> np.ndarray:
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

    frame = nuscenes.get('sample', frame_id)
    lidarseg_token = frame['data']['LIDAR_TOP']

    frame_annotations_lookup: set = set(frame['anns'])

    instance_annotations_lookup: set = \
        set(nuscenes.field2token('sample_annotation', 'instance_token', instance_id))

    intersection = list(set.intersection(frame_annotations_lookup, instance_annotations_lookup))

    assert len(intersection) == 1, \
        f"Frame {frame_id} should have the only instance of {instance_id}"

    annotation_token = intersection[0]

    _, boxes, _ = nuscenes.get_sample_data(lidarseg_token, selected_anntokens=[annotation_token])

    assert len(boxes) == 1
    box = boxes[0]

    points = frame_point_cloud[0:3, :]
    mask = points_in_box(box, points)

    annotation = nuscenes.get('sample_annotation', annotation_token)

    points_detected = np.sum(mask)
    points_expected = annotation['num_lidar_pts']
    assert points_expected == points_detected, \
        f"Expected {points_expected} points, detected {points_detected} points"

    instance_point_cloud = frame_point_cloud[:, np.where(mask)[0]]

    lidarseg_record = nuscenes.get('sample_data', lidarseg_token)

    # First step: transform the point cloud to the ego vehicle basis.
    calibrated_sensor_record = nuscenes.get('calibrated_sensor', lidarseg_record['calibrated_sensor_token'])
    point_cloud_to_ego_transformation = transform_matrix(np.array(calibrated_sensor_record['translation']),
                                                         Quaternion(calibrated_sensor_record['rotation']),
                                                         inverse=False)
    instance_point_cloud = __apply_transformation_matrix(point_cloud=instance_point_cloud,
                                                         transformation_matrix=point_cloud_to_ego_transformation)

    # Second step: transform from vehicle ego basis to the global basis.
    ego_pose_record = nuscenes.get('ego_pose', lidarseg_record['ego_pose_token'])
    ego_to_global_transformation = transform_matrix(np.array(ego_pose_record['translation']),
                                                    Quaternion(ego_pose_record['rotation']),
                                                    inverse=False)
    instance_point_cloud = __apply_transformation_matrix(point_cloud=instance_point_cloud,
                                                         transformation_matrix=ego_to_global_transformation)

    # Third step: reset object global transformation to identity rotation and translation.
    identity_transformation = transform_matrix(annotation['translation'],
                                               Quaternion(annotation['rotation']),
                                               inverse=True)
    instance_point_cloud = __apply_transformation_matrix(point_cloud=instance_point_cloud,
                                                         transformation_matrix=identity_transformation)

    return instance_point_cloud


def get_frame_point_cloud(frame_id: str,
                          nuscenes: NuScenes) -> np.ndarray:
    """

    :param frame_id:
    :param nuscenes:
    :return:
    """
    frame = nuscenes.get('sample', frame_id)
    lidarseg_token = frame['data']['LIDAR_TOP']
    lidarseg = nuscenes.get('sample_data', lidarseg_token)

    lidar_point_cloud = LidarPointCloud.from_file(os.path.join(nuscenes.dataroot, lidarseg['filename']))
    return lidar_point_cloud.points


def reapply_scene_transformation(annotation_token: str,
                                 lidarseg_token: str,
                                 point_cloud: np.ndarray,
                                 nuscenes: NuScenes) -> np.ndarray:
    annotation = nuscenes.get('sample_annotation', annotation_token)
    lidarseg_record = nuscenes.get('sample_data', lidarseg_token)

    # First step: put object back to the global coordinates.
    identity_to_global_transformation = transform_matrix(annotation['translation'],
                                                         Quaternion(annotation['rotation']),
                                                         inverse=False)
    instance_point_cloud = __apply_transformation_matrix(point_cloud=point_cloud,
                                                         transformation_matrix=identity_to_global_transformation)

    # Second step: transform from global to vehicle ego basis.
    ego_pose_record = nuscenes.get('ego_pose', lidarseg_record['ego_pose_token'])
    global_to_ego_transformation = transform_matrix(np.array(ego_pose_record['translation']),
                                                    Quaternion(ego_pose_record['rotation']),
                                                    inverse=True)
    instance_point_cloud = __apply_transformation_matrix(point_cloud=instance_point_cloud,
                                                         transformation_matrix=global_to_ego_transformation)

    # Third step: transform ego vehicle to the sensor basis.
    calibrated_sensor_record = nuscenes.get('calibrated_sensor', lidarseg_record['calibrated_sensor_token'])
    ego_to_sensor_transformation = transform_matrix(np.array(calibrated_sensor_record['translation']),
                                                    Quaternion(calibrated_sensor_record['rotation']),
                                                    inverse=True)
    instance_point_cloud = __apply_transformation_matrix(point_cloud=instance_point_cloud,
                                                         transformation_matrix=ego_to_sensor_transformation)

    return instance_point_cloud


def __apply_transformation_matrix(point_cloud: np.ndarray,
                                  transformation_matrix: np.ndarray) -> np.ndarray:
    """Applies given transformation matrix to the given point cloud.

    :param point_cloud: np.ndarray[float]
        Point cloud to transform.
    :param transformation_matrix: np.ndarray[float]
        Transformation matrix that describes rotation and translation.
    :return: np.ndarray[float]
        Modified point cloud.
    """
    points_count = point_cloud.shape[1]
    point_cloud[:3, :] = transformation_matrix.dot(
        np.vstack((point_cloud[:3, :], np.ones(points_count))))[:3, :]
    return point_cloud
