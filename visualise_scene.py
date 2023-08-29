import sys

from nuscenes.utils.data_classes import LidarPointCloud

from src.utils.visualisation_helper import visualise_points_cloud


def main():
    lidar_data_path = sys.argv[1]

    lidar_point_clouds = LidarPointCloud.from_file(lidar_data_path)
    visualise_points_cloud(lidar_point_clouds.points.T,
                           window_title=lidar_data_path)


if __name__ == '__main__':
    main()
