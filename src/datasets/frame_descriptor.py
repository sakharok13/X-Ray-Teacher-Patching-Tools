class FrameDescriptor(object):
    def __init__(self,
                 frame_id: str,
                 instances_ids: list):
        self.__frame_id = frame_id
        self.__instances_ids = instances_ids

    @property
    def frame_id(self) -> str:
        return self.__frame_id

    @property
    def instances_ids(self) -> list:
        return self.__instances_ids
