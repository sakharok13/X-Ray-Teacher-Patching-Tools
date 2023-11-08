from __future__ import annotations

from src.datasets.dataset import Dataset
from src.datasets.frame_descriptor import FrameDescriptor
from src.datasets.once.once_utils import ONCE
from src.datasets.once.once_utils import get_frame_ids_for_scene, get_pickle_data, aggregate_frames_in_sequences, \
    build_frame_id_to_annotations_lookup


class OnceSceneIterator(Dataset.SceneIterator):
    """Iterator over frames in Once scene.
    """

    def __init__(self,
                 scene_id: str,
                 once: ONCE):
        self.__scene_id = scene_id
        self.__once = once
        self.__frame_ids = get_frame_ids_for_scene(once=once,
                                                   scene_id=scene_id)

        # TODO: replace with constant runtime operation.
        self.__pickle_data = get_pickle_data(self.__once.dataset_root, self.__scene_id)
        self.__sequences_to_frames_lookup = aggregate_frames_in_sequences(self.__pickle_data)
        self.__frame_id_to_annotations_lookup = build_frame_id_to_annotations_lookup(self.__pickle_data)

        self.__current_index = 0

    def __iter__(self) -> OnceSceneIterator:
        """Reset iterator and returns itself.
        """
        self.__current_index = 0
        return self

    def __next__(self) -> tuple[str, FrameDescriptor]:
        """Returns next frame.

        :return: tuple[str, dict[str, any]]
            Returns a tuple of frame id to frame meta-information
        """

        if self.__current_index >= len(self.__frame_ids):
            raise StopIteration()

        frame_id = self.__frame_ids[self.__current_index]

        # Frame descriptor can be None if there are no detections for the frame.
        instance_ids = []

        # TODO: replace with constant runtime operation.
        if frame_id in self.__frame_id_to_annotations_lookup:
            frame_descriptor = self.__frame_id_to_annotations_lookup[frame_id]
            instance_ids = frame_descriptor['annos']['instance_ids']

        self.__current_index += 1

        return frame_id, FrameDescriptor(
            frame_id=frame_id, instances_ids=instance_ids)
