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
        self.__frame_ids = self.__get_frame_ids()
        self.__current_sample = self.__get_first_frame()
        self.__current_id = 0
        # self.__annotated_frames = self.__get_annotated_frames()

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

        frame_info = {}
        instance_ids = []

        frame_id = self.__current_sample
        anno_json = get_annotations_tracked_file_name(self.__once.data_folder, self.__scene)
        with open(anno_json, 'r') as json_file:
            data = json.load(json_file)

        while self.__current_id < len(self.__frame_ids):
            print("Iterating frame " + str(self.__current_id + 1) + " of " + str(len(self.__frame_ids)))
            if 'frames' in data and frame_id in {frame.get('frame_id') for frame in data['frames']}:
                frame_info = next(frame for frame in data['frames'] if frame.get('frame_id') == frame_id)
                if 'annos' in frame_info:
                    instance_ids = frame_info['annos']['instance_ids']
                    self.__current_id += 1
                    if self.__current_id < len(self.__frame_ids):
                        self.__current_sample = self.__frame_ids[self.__current_id]
                    break  # found annotations
            self.__current_id += 1
            if self.__current_id < len(self.__frame_ids):
                self.__current_sample = self.__frame_ids[self.__current_id]
                frame_id = self.__current_sample

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
        frame_ids = []
        frames_folder_path = os.path.join(self.__once.data_folder, self.__scene, 'lidar_roof')

        for file_name in os.listdir(frames_folder_path):
            match = re.search(r'\d+', file_name)
            if match:
                numeric_part = match.group()
                frame_ids.append(str(int(numeric_part)))

        return frame_ids
