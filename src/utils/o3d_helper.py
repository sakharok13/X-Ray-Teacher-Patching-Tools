import open3d as o3d
import numpy as np


def convert_to_o3d_pointcloud(
        points: np.ndarray[float]) -> o3d.geometry.PointCloud:
    """Converts a point cloud in numpy array format to an Open3D point cloud.

    :param points: np.ndarray
        Point cloud as a mxk float numpy array with columns X, Y, Z, intensity.
    :return: o3d.geometry.PointCloud
        Open3D point cloud object with intensity-based coloring.
    """
    xyz = points[:3, :].T
    intensity = points[3, :].reshape(-1, 1)

    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(xyz)

    colors = np.zeros((len(intensity), 3))
    colors[:, 0] = intensity

    o3d_pc.colors = o3d.utility.Vector3dVector(colors)

    return o3d_pc
