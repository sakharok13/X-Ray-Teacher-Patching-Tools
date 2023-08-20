import numpy as np
import open3d as o3d


def visualise_points_cloud(points: np.ndarray[float]):
    """Draws a point cloud in open3d.

    :param points: np.ndarray
        Point cloud as a mx3 float numpy array
    """

    vector3d = o3d.utility.Vector3dVector(points)
    o3d_pc = o3d.geometry.PointCloud(points=vector3d)
    o3d.visualization.draw_geometries([o3d_pc])
