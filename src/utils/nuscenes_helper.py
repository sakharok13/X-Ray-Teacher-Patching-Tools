import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box


def group_instances_across_frames(scene_id: int,
                                  nuscenes: NuScenes) -> dict[str, list[str]]:
    """ Returns a dictionary of instances associated with the frames which contain them.

    :param scene_id: int
        ID of a scene where grouping is required.
    :param nuscenes: 'NuScenes'
        NuScenes dataset facade.
    :return: dict[str, list[str]]
        Returns a dict of pairs of instance id to the list of ids of frames.
    """

    grouped_instances: dict[str, list[str]] = dict()

    scene = nuscenes.scene[scene_id]

    current_frame_token = scene['first_sample_token']

    while current_frame_token is not None and len(current_frame_token) > 0:
        frame = nuscenes.get('sample', current_frame_token)

        annotations = frame['anns']

        for annotation in annotations:
            annotation = nuscenes.get('sample_annotation', annotation)

            instance_id = annotation['instance_token']

            if instance_id not in grouped_instances:
                grouped_instances[instance_id] = list()

            grouped_instances[instance_id].append(current_frame_token)

        current_frame_token = frame['next']

    return grouped_instances


def get_instance_point_cloud(frame_id: str,
                             instance_id: str,
                             scene_point_cloud: LidarPointCloud,
                             nuscenes: NuScenes) -> np.ndarray[float]:
    """Returns point cloud for the given instance in the given frame.

    :param frame_id: str
        ID of a frame (aka sample).
    :param instance_id: str
        ID of an instance.
    :param nuscenes: 'NuScenes'
        NuScenes dataset facade.
    :param scene_point_cloud: 'LidarPointCloud'
        Point Cloud from lidar.
    :return: np.ndarray[float]
        Returns point cloud for the given object.
        Dimension of the array is 3xm.
    """

    frame = nuscenes.get('sample', frame_id)
    lidarseg_token = frame['data']['LIDAR_TOP']

    frame_annotations_lookup: set[str] = set(frame['anns'])

    instance_annotations_lookup: set[str] = \
        set(nuscenes.field2token('sample_annotation', 'instance_token', instance_id))

    intersection = list(set.intersection(frame_annotations_lookup, instance_annotations_lookup))

    assert len(intersection) == 1, \
        f"Frame {frame_id} should have the only instance of {instance_id}"

    annotation_token = intersection[0]

    _, boxes, _ = nuscenes.get_sample_data(lidarseg_token, selected_anntokens=[annotation_token])

    assert len(boxes) == 1
    box = boxes[0]

    points = scene_point_cloud.points[0:3, :]
    mask = points_in_box(box, points)

    annotation = nuscenes.get('sample_annotation', annotation_token)

    points_detected = np.sum(mask)
    points_expected = annotation['num_lidar_pts']
    assert points_expected == points_detected, \
        f"Expected {points_expected} points, detected {points_detected} points"

    return points[:, np.where(mask)[0]]
