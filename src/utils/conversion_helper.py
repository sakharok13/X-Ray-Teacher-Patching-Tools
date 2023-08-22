import open3d as o3d
import numpy as np

def convert_to_o3d_pointcloud(points: np.ndarray):
    """Converts a point cloud in numpy array format to an Open3D point cloud.

    :param points: np.ndarray
        Point cloud as a mxk float numpy array with columns X, Y, Z, intensity.
    :return: o3d.geometry.PointCloud
        Open3D point cloud object.
    """
    xyz = points[:, :3]

    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(xyz)

    return o3d_pc

