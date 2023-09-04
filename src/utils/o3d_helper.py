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

    # xyz = points[:3, :].T
    # intensity = points[3, :].reshape(-1, 1)

    if points.shape[0] == 1:
        xyz = points[:, :3].reshape(1, -1)
        intensity = points[:, 3].reshape(-1, 1)
    else:
        xyz = points[:, :3]
        intensity = points[:, 3]

    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(xyz)

    colors = np.zeros((len(intensity), 3))
    colors[:, 0] = intensity

    o3d_pc.colors = o3d.utility.Vector3dVector(colors)

    return o3d_pc


def convert_to_numpy_array(
        o3d_pc: o3d.geometry.PointCloud) -> np.ndarray[float]:
    """Converts an Open3D point cloud to a numpy array.

    :param o3d_pc: o3d.geometry.PointCloud
        Open3D point cloud object.
    :return: np.ndarray[float]
        Point cloud as a mxk float numpy array with columns X, Y, Z, intensity.
    """
    print("converting back: ")
    xyz = np.asarray(o3d_pc.points).T
    print(xyz)

    colors = np.asarray(o3d_pc.colors)
    intensity = colors[:, 0]

    point_cloud_np = np.vstack((xyz, intensity))  # .T
    print(point_cloud_np)

    return point_cloud_np
