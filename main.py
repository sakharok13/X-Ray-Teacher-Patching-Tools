from nuscenes import NuScenes

from src.utils.nuscenes_helper import group_instances_across_frames


def main():
    nuscenes = NuScenes(version='v1.0-mini', dataroot='./temp/nuscenes', verbose=True)

    grouped_instances = group_instances_across_frames(scene_id=0, nuscenes=nuscenes)

    print(grouped_instances)


if __name__ == '__main__':
    main()
