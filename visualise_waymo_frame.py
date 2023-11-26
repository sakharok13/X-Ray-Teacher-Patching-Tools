import argparse
import numpy as np

from src.utils.visualisation_helper import visualise_points_cloud


def parse_arguments():
    parser = argparse.ArgumentParser(description='patch scene arguments')
    parser.add_argument('file', type=str, help='Data root location.')
    parser.add_argument('--camera_position', type=str, default=None, help='Directory to save tracked files.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    lidar_data_path = args.file

    lidar_point_clouds = np.load(lidar_data_path)
    visualise_points_cloud(lidar_point_clouds,
                           camera_position=args.camera_position,
                           window_title=lidar_data_path)


if __name__ == '__main__':
    main()
