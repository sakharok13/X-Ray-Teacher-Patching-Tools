import numpy as np
import open3d as o3d
from nuscenes import NuScenes


class SceneInstance:
    def __init__(self, instance_id: str, category: str, scene_id: int):
        self.instance_id = instance_id
        self.category = category
        self.scene_id = scene_id
        self.accumulated_points = []  # 3d point clouds
        self.frames = []  # list of frames where the object is contained
        self.transformations = []  # list of transformations to be applied to the object to move it to each frame

    def add_frame_data(self, scene_id: int, frame_id: str, transformation_matrix: np.ndarray):
        self.scene_id = scene_id
        self.frames.append(frame_id)
        self.transformations.append(transformation_matrix)

    def reset_accumulated_points(self):
        self.accumulated_points = []

    def __str__(self):
        return f"Instance ID: {self.instance_id}, Category: {self.category}, " \
               f"Scene ID: {self.scene_id}, Frames: {self.frames}, " \
               f"Transformations: {len(self.transformations)}"

    def __repr__(self):
        return self.__str__()