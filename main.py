import os.path

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from src.utils.nuscenes_helper import group_instances_across_frames, get_instance_point_cloud
from src.utils.visualisation_helper import visualise_points_cloud


def main():
    nuscenes = NuScenes(version='v1.0-mini', dataroot='./temp/nuscenes', verbose=True)

    grouped_instances = group_instances_across_frames(scene_id=0, nuscenes=nuscenes)

    frame = nuscenes.get('sample', 'e0845f5322254dafadbbed75aaa07969')
    lidarseg_token = frame['data']['LIDAR_TOP']
    lidarseg = nuscenes.get('sample_data', lidarseg_token)

    scene_point_cloud = LidarPointCloud.from_file(os.path.join(nuscenes.dataroot, lidarseg['filename']))

    points = get_instance_point_cloud(frame_id='e0845f5322254dafadbbed75aaa07969',
                                      instance_id='e91afa15647c4c4994f19aeb302c7179',
                                      scene_point_cloud=scene_point_cloud,
                                      nuscenes=nuscenes)

    visualise_points_cloud(points.T)

    print(grouped_instances)


if __name__ == '__main__':
    main()
