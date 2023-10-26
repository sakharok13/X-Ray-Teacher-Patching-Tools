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


def can_skip_scene(dataset: Dataset,
                   scene_id: str,
                   force_overwrite: bool) -> bool:
    scene_iterator = dataset.get_scene_iterator(scene_id=scene_id)
    for frame_id, frame in scene_iterator:
        if not can_skip_frame(dataset=dataset,
                              scene_id=scene_id,
                              frame_id=frame_id,
                              force_overwrite=force_overwrite):
            return False
    return True


def can_skip_frame(dataset: Dataset,
                   scene_id: str,
                   frame_id: str,
                   force_overwrite: bool) -> bool:
    return not force_overwrite and not dataset.can_serialise_frame_point_cloud(scene_id=scene_id,
                                                                               frame_id=frame_id)
