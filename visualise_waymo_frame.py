import sys
import numpy as np

from src.utils.visualisation_helper import visualise_points_cloud


def main():
    lidar_data_path = sys.argv[1]

    lidar_point_clouds = np.load(lidar_data_path)
    visualise_points_cloud(lidar_point_clouds.points.T,
                           window_title=lidar_data_path)


if __name__ == '__main__':
    main()
