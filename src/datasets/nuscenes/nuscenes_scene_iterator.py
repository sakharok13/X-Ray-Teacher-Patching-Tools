from __future__ import annotations

from nuscenes import NuScenes

from src.datasets.dataset import Dataset
from src.datasets.frame_descriptor import FrameDescriptor


class NuScenesSceneIterator(Dataset.SceneIterator):
    """Iterator over frames in a NuScenes scene.
    """

    def __init__(self,
                 scene: dict,
                 nuscenes: NuScenes):
        self.__scene = scene
        self.__nuscenes = nuscenes
        self.__current_sample = self.__get_first_frame()

    def __iter__(self) -> NuScenesSceneIterator:
        """Reset iterator and returns itself.
        """
        self.__current_sample = self.__get_first_frame()
        return self

    def __next__(self) -> tuple[str, FrameDescriptor]:
        """Returns next frame.

        :return: tuple[str, dict[str, any]]
            Returns a tuple of frame id to frame meta-information
        """

        if self.__current_sample is None or len(self.__current_sample) == 0:
            raise StopIteration()

        frame_id = self.__current_sample
        # NuScenes guarantees to return a record in constant runtime.
        raw_frame = self.__nuscenes.get('sample', frame_id)

        # TODO: replace with constant runtime operation.
        instance_ids = [self.__nuscenes.get('sample_annotation', annotation)['instance_token'] for annotation in
                        raw_frame['anns']]

        self.__current_sample = raw_frame['next']

        return frame_id, FrameDescriptor(frame_id=frame_id, instances_ids=instance_ids)

    def __get_first_frame(self) -> str:
        """Returns first frame id.

        :return: str
            First frame ID as a string.
        """
        return self.__scene['first_sample_token']
