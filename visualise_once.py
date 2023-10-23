import sys
import numpy as np

from src.utils.visualisation_helper import visualise_points_cloud


def main():
    lidar_data_path = sys.argv[1]

    lidar_point_clouds = np.fromfile(lidar_data_path, dtype=np.float32).reshape(-1, 4).T
    visualise_points_cloud(lidar_point_clouds.T,
                           window_title=lidar_data_path)


if __name__ == '__main__':
    main()
