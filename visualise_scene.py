import sys

from nuscenes.utils.data_classes import LidarPointCloud

from src.utils.visualisation_helper import visualise_points_cloud


def main():
    path = sys.argv[1]
    print(sys.argv)

    lidar_point_clouds = LidarPointCloud.from_file(path)
    visualise_points_cloud(lidar_point_clouds.points.T)


if __name__ == '__main__':
    main()
