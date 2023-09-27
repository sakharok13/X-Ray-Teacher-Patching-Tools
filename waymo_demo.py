import numpy as np

from src.datasets.waymo.waymo_utils import get_frame_point_cloud, load_scene_descriptor, get_instance_point_cloud
from src.datasets.waymo.waymo_dataset import WaymoDataset
from src.utils.visualisation_helper import visualise_points_cloud

if __name__ == '__main__':
    dataset = WaymoDataset(dataset_root='./temp/open-waymo')

    print(dataset.scenes)

    scene_descriptor = load_scene_descriptor(dataset_root='./temp/open-waymo',
                                             scene_id='training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels')

    frame_id = 'training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels_010'
    frame_descriptor = scene_descriptor[frame_id]

    frame_point_cloud = get_frame_point_cloud(dataset_root='./temp/open-waymo',
                                              scene_id='training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels',
                                              frame_descriptor=frame_descriptor)

    visualise_points_cloud(frame_point_cloud.T)

    annotations = frame_descriptor['annotations']

    print('instances', annotations['ids'])

    for id in annotations['ids']:
        instance_point_cloud = get_instance_point_cloud(frame_point_cloud=frame_point_cloud,
                                                        instance_id=id,
                                                        frame_descriptor=frame_descriptor)

        visualise_points_cloud(instance_point_cloud.T, window_title=id)


