import argparse
import sys
import numpy as np

from src.utils.visualisation_helper import visualise_points_cloud


def find_annotations(frame_file: str,
                     frame_descriptors: list) -> dict:
    for frame_descriptor in frame_descriptors:
        if frame_descriptor['frame_id'] in frame_file:
            return frame_descriptor

    raise ValueError('Cannot fine suitable annotations')


def parse_arguments():
    parser = argparse.ArgumentParser(description='patch scene arguments')
    parser.add_argument('file', type=str, help='Data root location.')
    parser.add_argument('--frame_descriptor', type=str, default=None, help='Data root location.')
    parser.add_argument('--camera_position', type=str, default=None, help='Directory to save tracked files.')
    return parser.parse_args()


def main():
    args = parse_arguments()

    lidar_data_path = args.file
    frame_descriptor_pickle_path = args.frame_descriptor
    camera_position = args.camera_position

    if frame_descriptor_pickle_path is not None:
        frame_descriptors = np.load(frame_descriptor_pickle_path, allow_pickle=True)
        frame_descriptor = find_annotations(frame_file=lidar_data_path, frame_descriptors=frame_descriptors)
        bboxes = frame_descriptor['annos']['boxes_3d']
    else:
        bboxes = []

    lidar_point_clouds = np.fromfile(lidar_data_path, dtype=np.float32).reshape(-1, 4).T
    visualise_points_cloud(lidar_point_clouds.T,
                           bboxes=bboxes,
                           camera_position=camera_position,
                           window_title=lidar_data_path)


if __name__ == '__main__':
    main()
