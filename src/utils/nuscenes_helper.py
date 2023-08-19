import numpy as np
from nuscenes import NuScenes


def group_instances_across_frames(scene_id: int,
                                  nuscenes: NuScenes) -> dict[str, list[str]]:
    """ Returns a dictionary of instances associated with the frames which contain them.

    :param scene_id: int
        ID of a scene where grouping is required.
    :param nuscenes: 'NuScenes'
        NuScenes dataset facade.
    :return: dict[str, list[str]]
        Returns a dict of pairs of instance id to the list of ids of frames.
    """

    grouped_instances: dict[str, list[str]] = dict()

    scene = nuscenes.scene[scene_id]

    current_frame_token = scene['first_sample_token']

    while current_frame_token is not None and len(current_frame_token) > 0:
        frame = nuscenes.get('sample', current_frame_token)

        annotations = frame['anns']

        for annotation in annotations:
            annotation = nuscenes.get('sample_annotation', annotation)

            instance_id = annotation['instance_token']

            if instance_id not in grouped_instances:
                grouped_instances[instance_id] = list()

            grouped_instances[instance_id].append(current_frame_token)

        current_frame_token = frame['next']

    return grouped_instances
