import numpy as np
import open3d as o3d

from nuscenes import NuScenes
from typing import Optional

from src.datasets.bbox import Bbox
from src.utils.conversion_helper import convert_to_o3d_pointcloud


class SceneInstance:
    def __init__(self,
                 instance_id: str,
                 scene_id: int,
                 raw_point_cloud: np.ndarray,
                 bbox: Optional[Bbox],
                 category: Optional[str]):
        self.__instance_id = instance_id
        self.__category = category
        self.__bbox = bbox
        self.__scene_id = scene_id
        self.__raw_point_cloud = raw_point_cloud

    @property
    def id(self) -> str:
        return self.__instance_id

    @property
    def category(self) -> Optional[str]:
        return self.__category

    @property
    def scene_id(self) -> int:
        return self.__scene_id

    @property
    def raw_point_cloud(self) -> np.ndarray:
        return self.__raw_point_cloud

    @property
    def bbox(self) -> Optional[Bbox]:
        return self.__bbox


class FrameDetection(SceneInstance):
    def __init__(self, instance_id: str, category: str, scene_id: int, frame_token: str,
                 transformation_matrix: np.ndarray):
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
