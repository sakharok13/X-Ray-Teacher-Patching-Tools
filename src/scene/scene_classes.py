import numpy as np
import open3d as o3d
from nuscenes import NuScenes
from src.utils.o3d_helper import convert_to_o3d_pointcloud

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


class FrameDetection(SceneInstance):
    """
        Represents a singular detection of scene instances in a single frame.

        This is a temporary class used to store information about scene instances
        detected in a single frame of data.

        Attributes:
            (list any attributes specific to this class here)
        """
    def __init__(self, instance_id: str, category: str, scene_id: int, frame_token: str, transformation_matrix: np.ndarray):
        # Call the constructor of the parent class
        super().__init__(instance_id, category, scene_id)

        # Add attributes specific to frame detections
        self.frame_token = frame_token
        self.transformation_matrix = transformation_matrix
        self.points = []  # Points specific to this frame detection
        self.o3d_pointcloud = None  # Open3D point cloud representation

    def add_points(self, points: np.ndarray):
        self.points.extend(points)

    def create_o3d_pointcloud(self):
        if self.points:
            self.o3d_pointcloud = convert_to_o3d_pointcloud(self.points)

    def __str__(self):
        return f"Frame Token: {self.frame_token}, {super().__str__()}"

    def __repr__(self):
        return self.__str__()
