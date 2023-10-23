import numpy as np
import open3d as o3d


from pyquaternion import Quaternion


def visualise_points_cloud(points: np.ndarray,
                           bboxes: list = [],
                           window_title: str = 'Scene visualisation'):
    """Draws a point cloud in open3d.

    :param points: np.ndarray[float]
        Point cloud as a mxk float numpy array.
    :param bboxes: list[list[float]]
        List of bounding boxes in [cx, cy, cz, dx, dy, dz, heading] format.
    :param window_title: str
        Title of the window.
    """

    vector3d = o3d.utility.Vector3dVector(points[:, 0:3])
    o3d_pc = o3d.geometry.PointCloud(points=vector3d)

    o3d_boxes = []

    for bbox in bboxes:
        x, y, z, dx, dy, dz, heading = bbox
        center = (x, y, z)
        r = Quaternion(heading, 0, 0, 1).rotation_matrix
        size = np.array([dx, dy, dz])
        oriented_box = o3d.geometry.OrientedBoundingBox(center, r, size)
        o3d_boxes.append(oriented_box)

    o3d.visualization.draw_geometries([o3d_pc] + o3d_boxes,
                                      window_name=window_title)
