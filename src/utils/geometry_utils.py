import numpy as np
from pyquaternion import Quaternion
import math
from shapely.geometry import Polygon

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


def __corners(centers_xyz: np.ndarray,
              sizes_lwh: np.ndarray,
              orientation: Quaternion) -> np.ndarray:
    """
    Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    l, w, h = sizes_lwh

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(orientation.rotation_matrix, corners)

    # Translate
    x, y, z = centers_xyz
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


def points_in_box(center_xyz: np.ndarray,
                  dimensions_lwh: np.ndarray,
                  heading_angle: float,
                  points: np.ndarray):
    """
    Checks whether points are inside the box.

    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param box: <Box>.
    :param points: <np.float: 3, n>.
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >.
    """

    orientation = Quaternion(heading_angle, 0, 0, 1)
    corners = __corners(center_xyz, dimensions_lwh, orientation)

    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]
    p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v)
    jv = np.dot(j, v)
    kv = np.dot(k, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    return mask


def apply_transformation_matrix(point_cloud: np.ndarray,
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


def project_box_to_corners2d(box):
    cx, cy, cz, l, w, h, theta = box

    hl = l / 2
    hw = w / 2

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    corners_2d = [
        (cx - hl * cos_theta - hw * sin_theta, cy - hl * sin_theta + hw * cos_theta),
        (cx + hl * cos_theta - hw * sin_theta, cy + hl * sin_theta + hw * cos_theta),
        (cx + hl * cos_theta + hw * sin_theta, cy + hl * sin_theta - hw * cos_theta),
        (cx - hl * cos_theta + hw * sin_theta, cy - hl * sin_theta - hw * cos_theta)
    ]

    return corners_2d
def intersection_area(rect1, rect2):
    polygon1 = Polygon(rect1)
    polygon2 = Polygon(rect2)
    intersection = polygon1.intersection(polygon2)

    if intersection.is_empty:
        return 0.0

    return intersection.area

def calculate_iou(box1, box2):
    """Calculates fast estimation IOU for two ONCE-format boxes
    assuming the boxes don't differ in height too much and are
    aligned on Z axis
    :param box1: cx, cy, cz, l, w, h, theta
    :param box2: cx, cy, cz, l, w, h, theta
    """
    cx1, cy1, cz1, l1, w1, h1, theta1 = box1
    cx2, cy2, cz2, l2, w2, h2, theta2 = box2

    # calculate intersection on 2d projections of boxes for faster estimation
    rect1 = project_box_to_corners2d(box1)
    rect2 = project_box_to_corners2d(box2)

    int_area = intersection_area(rect1, rect2)

    h = (h1+h2)/2

    intersection = int_area * h

    volume1 = l1 * w1 * h1
    volume2 = l2 * w2 * h2

    union = volume1 + volume2 - intersection
    iou = intersection / union

    return iou
