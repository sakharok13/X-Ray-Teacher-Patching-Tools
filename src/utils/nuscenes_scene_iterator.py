from __future__ import annotations

from nuscenes import NuScenes


class NuScenesSceneIterator:
    """Iterator over frames in a NuScenes scene.
    """

    def __init__(self,
                 scene_id: int,
                 nuscenes: NuScenes):
        self.__scene_id = scene_id
        self.__nuscenes = nuscenes
        self.__scene = nuscenes.scene[scene_id]
        self.__current_sample = self.__get_first_frame()

    def __iter__(self) -> NuScenesSceneIterator:
        """Reset iterator and returns itself.
        """
        self.__current_sample = self.__get_first_frame()
        return self

    def __next__(self) -> tuple[str, dict[str, any]]:
        """Returns next frame.

        :return: tuple[str, dict[str, any]]
            Returns a tuple of frame id to frame meta-information
        """

        if self.__current_sample is None or len(self.__current_sample) == 0:
            raise StopIteration()

        frame_id = self.__current_sample
        frame = self.__nuscenes.get('sample', frame_id)

        self.__current_sample = frame['next']

        return frame_id, frame

    def __get_first_frame(self) -> str:
        """Returns first frame id.

        :return: str
            First frame ID as a string.
        """
        return self.__scene['first_sample_token']
