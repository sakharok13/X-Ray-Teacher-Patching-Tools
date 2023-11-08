from src.datasets.dataset import Dataset


def group_instances_across_frames(scene_id: str,
                                  dataset: Dataset) -> dict:
    """ Returns a dictionary of instances associated with the frames which contain them.

    Runtime complexity is O(frames * instances).

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
    """Checks whether it is possible to skip a scene.

    Scene is considered as skippable if and only if
    it is possible to skip all the frames within the
    scene.

    Runtime complexity is O(frames).

    :param dataset: Dataset
        Dataset to which the scene belongs to.
    :param scene_id: str
        Unique scene identifier.
    :param force_overwrite: bool
        Flag that forces a frame to be non-skippable if
        True.
    :return:
        True if can skip scene handling and False otherwise.
    """

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
    """ Checks whether it is possible to skip the frame.

    Frame is skippable if force overwrite flag is True
    or there is no point cloud saved on disk.

    Runtime complexity is O(1).

    :param dataset: Dataset
        Dataset to which the scene and the frame belongs to.
    :param scene_id: str
        Unique scene identifier.
    :param frame_id: str
        Unique frame identifier.
    :param force_overwrite: bool
        Flag that forces a frame to be non-skippable if
        True.
    :return:
        True if can skip frame handling and False otherwise.
    """
    return not force_overwrite and not dataset.can_serialise_frame_point_cloud(scene_id=scene_id,
                                                                               frame_id=frame_id)
