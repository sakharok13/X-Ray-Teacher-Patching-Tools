import os
import sys
import open3d as o3d

import numpy as np
import tensorflow.compat.v1 as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

tf.logging.set_verbosity(tf.logging.ERROR)
tf.enable_eager_execution()


def visualise_points_cloud(points: np.ndarray):
    """Draws a point cloud in open3d.

    :param points: np.ndarray
        Point cloud as a mxk float numpy array.
    """

    vector3d = o3d.utility.Vector3dVector(points[:, 0:3])
    o3d_pc = o3d.geometry.PointCloud(points=vector3d)
    o3d.visualization.draw_geometries([o3d_pc])


class WaymoDataset:
    def __init__(self,
                 path: str):
        self.__path = path
        self.__dataset = tf.data.TFRecordDataset(path, compression_type='')

    def __extract_point_cloud(self,
                              frame: open_dataset.Frame) -> np.ndarray:
        range_images, camera_projections, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(
            frame)
        frame.lasers.sort(key=lambda laser: laser.name)

        # Camera projection point cloud is ignored.
        point_cloud, _ = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections,
                                                                        range_image_top_pose)

        visualise_points_cloud(point_cloud[0])

        # for laser_label in frame.laser_labels:
        #     center_x = laser_label.box.center_x
        #     center_y = laser_label.box.center_y
        #     center_z = laser_label.box.center_z
        #     width = laser_label.box.width
        #     length = laser_label.box.length
        #     heading = laser_label.box.heading
        #     speed_x = laser_label.metadata.speed_x
        #     speed_y = laser_label.metadata.speed_y
        #     accel_x = laser_label.metadata.accel_x
        #     accel_y = laser_label.metadata.accel_y
        #     label_type = laser_label.type
        #     obj_id = laser_label.id
        #     num_points = laser_label.num_lidar_points_in_box

    # Function call to extract LiDAR images
    def extract(self):
        for data in self.__dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            self.__extract_point_cloud(frame)


if __name__ == '__main__':
    path = sys.argv[1]

    dataset = WaymoDataset(path=path)
    dataset.extract()
