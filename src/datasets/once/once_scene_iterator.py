from __future__ import annotations
import os
import re
import json

from src.datasets.dataset import Dataset
from src.datasets.frame_descriptor import FrameDescriptor
from src.datasets.once.once_utils import ONCE
from src.datasets.once.once_utils import get_annotations_tracked_file_name

class OnceSceneIterator(Dataset.SceneIterator):
    """Iterator over frames in Once scene.
    """

    def __init__(self,
                 scene: str,
                 once: ONCE):
        self.__scene = scene
        self.__once = once
        self.__current_sample = self.__get_first_frame()
        self.__current_id = 0
        self.__frame_ids = self.__get_frame_ids()

    def __iter__(self) -> OnceSceneIterator:
        """Reset iterator and returns itself.
        """
        self.__current_sample = self.__get_first_frame()
        return self

    def __next__(self) -> tuple[str, FrameDescriptor]:
        """Returns next frame.

        :return: tuple[str, dict[str, any]]
            Returns a tuple of frame id to frame meta-information
        """

        if self.__current_sample is None or self.__current_id >= len(self.__frame_ids):
            raise StopIteration()

        frame_id = self.__current_sample
        anno_json = get_annotations_tracked_file_name(self.__once.data_root, self.__scene)
        with open(anno_json, 'r') as json_file:
            data = json.load(json_file)

        instance_ids = data["frames"][frame_id]["annos"]["instance_ids"]

        self.__current_id += 1
        self.__current_sample = self.__frame_ids[self.__current_id]

        return frame_id, FrameDescriptor(
            frame_id=frame_id, instances_ids=instance_ids)

    def __get_first_frame(self) -> str:
        """Returns first frame id.

        :return: str
            First frame ID as a string.
        """
        return self.__frame_ids[0]

    def __get_frame_ids(self):
        """Returns a list of frame IDs.

        :return: list[str]
            List of frame IDs as strings.
        """
        dataroot = self.__once.data_root
        frame_ids = []
        scene_folder_path = os.path.join(dataroot, 'data', self.__scene)

        for folder_name in os.listdir(scene_folder_path):
            match = re.search(r'\d+', folder_name)
            if match:
                numeric_part = match.group()
                frame_ids.append(str(int(numeric_part)))

        return frame_ids
