import numpy as np
import open3d as o3d

from pyquaternion import Quaternion
from typing import Optional


def visualise_points_cloud(points: np.ndarray,
                           bboxes: list = [],
                           camera_position: Optional = None,
                           point_size: float = 2,
                           window_title: str = 'Scene visualisation'):
    """Draws a point cloud in open3d.

    :param points: np.ndarray[float]
        Point cloud as a mxk float numpy array.
    :param bboxes: list[list[float]]
        List of bounding boxes in [cx, cy, cz, dx, dy, dz, heading] format.
    :param camera_position: Optional[str]
        Position of a camera obtained by pressing P in o3d window.
    :param point_size: float
        Size of a single point in the window.
    :param window_title: str
        Title of the window.
    """

    n = points.shape[0]

    vector3d = o3d.utility.Vector3dVector(points[:, 0:3])
    o3d_pc = o3d.geometry.PointCloud(points=vector3d)
    o3d_pc.colors = o3d.utility.Vector3dVector(np.zeros((n, 3)))

    o3d_boxes = []

    for bbox in bboxes:
        x, y, z, dx, dy, dz, heading = bbox
        center = (x, y, z)
        r = Quaternion(heading, 0, 0, 1).rotation_matrix
        size = np.array([dx, dy, dz])
        oriented_box = o3d.geometry.OrientedBoundingBox(center, r, size)
        o3d_boxes.append(oriented_box)

    def capture_image(vis):
        image_path = f"{window_title}.jpg"
        vis.capture_screen_image(image_path)
        print(f"Scene saved to: {image_path}")
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_title)

    vis.register_key_callback(ord("S"), capture_image)

    render_option = vis.get_render_option()
    render_option.point_size = point_size

    vis.add_geometry(o3d_pc)
    for box in o3d_boxes:
        vis.add_geometry(box)

    if camera_position is not None:
        control = vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters(camera_position)
        control.convert_from_pinhole_camera_parameters(parameters)
        vis.update_renderer()

    vis.run()
    vis.destroy_window()
