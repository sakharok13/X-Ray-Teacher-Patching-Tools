from src.datasets.dataset import Dataset


def group_instances_across_frames(scene_id: str,
                                  dataset: Dataset) -> dict:
    """ Returns a dictionary of instances associated with the frames which contain them.

    :param scene_id: int
        ID of a scene where grouping is required.
    :param dataset: 'Dataset'
        Dataset.
    :return: dict[str, list[str]]
        Returns a dict of pairs of instance id to the list of ids of frames.
    """

    grouped_instances = dict()

    scene_iterator = dataset.get_scene_iterator(scene_id=scene_id)

    for frame_id, frame in scene_iterator:
        instances_ids = frame.instances_ids

        for instance_id in instances_ids:
            if instance_id not in grouped_instances:
                grouped_instances[instance_id] = list()

            grouped_instances[instance_id].append(frame_id)

    return grouped_instances
