import sys
import numpy as np

from src.utils.visualisation_helper import visualise_points_cloud


def find_annotations(frame_file: str,
                     frame_descriptors: list) -> dict:
    for frame_descriptor in frame_descriptors:
        if frame_descriptor['frame_id'] in frame_file:
            return frame_descriptor

    raise ValueError('Cannot fine suitable annotations')

def main():
    lidar_data_path = sys.argv[1]

    if len(sys.argv) > 2:
        pickle_data_path = sys.argv[2]
        frame_descriptors = np.load(pickle_data_path, allow_pickle=True)
        frame_descriptor = find_annotations(frame_file=lidar_data_path, frame_descriptors=frame_descriptors)
        bboxes = frame_descriptor['annos']['boxes_3d']
    else:
        bboxes = []

    print(bboxes)

    lidar_point_clouds = np.fromfile(lidar_data_path, dtype=np.float32).reshape(-1, 4).T
    visualise_points_cloud(lidar_point_clouds.T,
                           bboxes=bboxes,
                           window_title=lidar_data_path)


if __name__ == '__main__':
    main()
