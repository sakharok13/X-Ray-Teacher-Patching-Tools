import json
import os
import pickle
from collections import defaultdict

import cv2
import numpy as np


class ONCE(object):
    """
    dataset structure:
    - data_root
        -ImageSets
            - train_split.txt
            - val_split.txt
            - test_split.txt
            - raw_split.txt
        - data
            - seq_id
                - cam01
                - cam03
                - ...
                -
    """

    supported_splits = [
        'raw_small'
        # 'raw_medium',
        # 'raw_large',
        # 'train',
        # 'test',
        # 'val',
    ]

    camera_names = [
        'cam01',
        'cam03',
        'cam05',
        'cam06',
        'cam07',
        'cam08',
        'cam09'
    ]

    camera_tags = [
        'top',
        'top2',
        'left_back',
        'left_front',
        'right_front',
        'right_back',
        'back'
    ]

    def __init__(self,
                 dataset_root: str,
                 scenes_root: str,
                 split: str):
        self.dataset_root = dataset_root
        self.data_folder = scenes_root

        self.__scenes_by_split_lookup = {s: self.__load_scenes_in_split(s) for s in self.supported_splits}

        self.__collect_basic_infos([split])

    def get_scenes_in_split(self, split: str) -> set:
        assert split in self.supported_splits
        return self.__scenes_by_split_lookup[split]

    def __load_scenes_in_split(self, split: str) -> set:
        assert split in self.supported_splits

        split_file = os.path.join(self.dataset_root, 'ImageSets', f"{split}.txt")

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Expected split descriptor at {split_file} but non was found")

        if not os.path.isfile(split_file):
            raise Exception(f"Not a file {split_file}")

        return set(map(lambda x: x.strip(), open(split_file).readlines()))

    def __find_split_name(self, scene_id: str):
        for split in self.supported_splits:
            if scene_id in self.__scenes_by_split_lookup[split]:
                return split

        raise Exception(f"sequence id {scene_id} corresponding to no split")

    def __collect_basic_infos(self, splits: list):
        self.train_info = defaultdict(dict)
        self.val_info = defaultdict(dict)
        self.test_info = defaultdict(dict)
        self.raw_small_info = defaultdict(dict)
        self.raw_medium_info = defaultdict(dict)
        self.raw_large_info = defaultdict(dict)

        for split in splits:
            assert split in self.supported_splits, \
                f"Unsupported split {split}"

            split_list = self.get_scenes_in_split(split)
            info_dict = getattr(self, f"{split}_info")
            for seq in split_list:
                anno_file_path = os.path.join(self.data_folder, seq, f"{seq}.json")
                if not os.path.isfile(anno_file_path):
                    raise FileNotFoundError(f"no annotation file for sequence {seq}, looked for {anno_file_path}")
                anno_file = json.load(open(anno_file_path, 'r'))
                frame_list = list()
                for frame_anno in anno_file['frames']:
                    frame_list.append(str(frame_anno['frame_id']))
                    info_dict[seq][frame_anno['frame_id']] = {
                        'pose': frame_anno['pose'],
                    }
                    info_dict[seq][frame_anno['frame_id']]['calib'] = dict()
                    for cam_name in self.__class__.camera_names:
                        info_dict[seq][frame_anno['frame_id']]['calib'][cam_name] = {
                            'cam_to_velo': np.array(anno_file['calib'][cam_name]['cam_to_velo']),
                            'cam_intrinsic': np.array(anno_file['calib'][cam_name]['cam_intrinsic']),
                            'distortion': np.array(anno_file['calib'][cam_name]['distortion'])
                        }
                    if 'annos' in frame_anno.keys():
                        info_dict[seq][frame_anno['frame_id']]['annos'] = frame_anno['annos']
                info_dict[seq]['frame_list'] = sorted(frame_list)

    def get_frame_point_cloud(self,
                              scene_id: str,
                              frame_id: str):
        bin_path = os.path.join(self.data_folder, scene_id, 'lidar_roof', f"{frame_id}.bin")
        return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4).T

    def move_back_to_frame_coordinates(self, points, box):
        cx, cy, cz, l, w, h, theta = box
        points = points.T

        # rotation transform matrix (rotate to plus! theta)
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])

        # to frame coordinates
        translated_points = points[:, :3] + np.array([cx, cy, cz])

        rotated_points = np.dot(translated_points, R.T)
        intensity = points[:, 3]
        transformed_points = np.column_stack((rotated_points, intensity))

        return transformed_points.T


def get_frame_instance_ids(scene_id, frame_id, once, frame_id_to_annotations_lookup={}):
    instance_ids = []

    pickle_data = get_pickle_data(once.dataset_root, scene_id)

    if frame_id_to_annotations_lookup == {}:
        frame_id_to_annotations_lookup = build_frame_id_to_annotations_lookup(pickle_data)

    if frame_id in frame_id_to_annotations_lookup:
        if 'annos' in frame_id_to_annotations_lookup[frame_id]:
            instance_ids = frame_id_to_annotations_lookup[frame_id]['annos']['instance_ids']

    return instance_ids


def get_instance_point_cloud(
        seq_id,
        frame_id,
        instance_id,
        frame_point_cloud,
        once):
    """Returns point cloud for the given instance in the given frame.

        The returned point cloud has reset rotation and translation.

        :param seq_id: str
            ID of a scene (sequence).
        :param frame_id: str
            ID of a frame (aka sample).
        :param instance_id: str
            ID of an instance.
        :param frame_point_cloud:
            np.ndarray point cloud.
        :param once:
            Once dataset instance.
        :return: np.ndarray[float]
            Returns point cloud for the given object.
        """

    pickle_data = get_pickle_data(once.dataset_root, seq_id)
    frame_id_to_annotations_lookup = build_frame_id_to_annotations_lookup(pickle_data)

    instance_ids = get_frame_instance_ids(seq_id, frame_id, once,
                                          frame_id_to_annotations_lookup=frame_id_to_annotations_lookup)

    annotations = frame_id_to_annotations_lookup[frame_id]['annos']

    if instance_id in instance_ids:
        box_index = instance_ids.index(instance_id)
        box = annotations['boxes_3d'][box_index]
        cx, cy, cz, l, w, h, theta = box
        maxd = max(l, w, h)

        frame_point_cloud_t = frame_point_cloud.T
        points = frame_point_cloud_t[:, :3]

        lower_bounds = [cx - maxd, cy - maxd, cz - maxd]
        upper_bounds = [cx + maxd, cy + maxd, cz + maxd]

        mask = np.all((lower_bounds <= points) & (points <= upper_bounds), axis=1)
        points_inside = frame_point_cloud_t[mask]

        reset_cloud = transform_points(points_inside, cx, cy, cz, l, w, h, theta)
        return reset_cloud.T
    else:
        raise ValueError(
            f"Instance ID {instance_id} is not present in the instance_ids list.")


def transform_points(points, cx, cy, cz, l, w, h, theta):
    theta_rad = -theta  # rotation transform matrix (rotate to minus! theta)
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                  [np.sin(theta_rad), np.cos(theta_rad), 0],
                  [0, 0, 1]])

    points_xyz = points[:, :3]

    translated_points = points_xyz - np.array([cx, cy, cz])
    rotated_points = np.dot(translated_points, R.T)

    inside_mask = (np.all(np.logical_and((-l / 2 <= rotated_points), (rotated_points <= l / 2)), axis=1) &
                   np.all(np.logical_and((-w / 2 <= rotated_points), (rotated_points <= w / 2)), axis=1) &
                   np.all(np.logical_and((-h / 2 <= rotated_points), (rotated_points <= h / 2)), axis=1))
    intensity = points[inside_mask, 3]
    transformed_points = rotated_points[inside_mask]
    points_inside = np.column_stack((transformed_points, intensity))

    return points_inside


def reapply_frame_transformation(point_cloud: np.ndarray,
                                 frame_descriptor: dict,
                                 instance_id: str,
                                 once: ONCE) -> np.ndarray:
    annotations = frame_descriptor['annos']
    ids = annotations['instance_ids']
    instance_index = ids.index(instance_id)
    boxes = annotations['boxes_3d']
    box = boxes[instance_index]

    moved_cloud = once.move_back_to_frame_coordinates(point_cloud, box)

    return moved_cloud


def get_pickle_data(dataset_root, scene_id):
    root_files = [f for f in os.listdir(dataset_root) if f.endswith(".pkl")]
    scene_path = os.path.join(dataset_root, 'data', scene_id)

    for root_file in root_files:
        if scene_id in root_file:
            pickle_path = os.path.join(dataset_root, root_file)
            with open(pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            return pickle_data

    if os.path.exists(scene_path):
        scene_files = [f for f in os.listdir(scene_path) if f.endswith(".pkl")]
        if scene_files:
            pickle_path = os.path.join(scene_path, scene_files[0])
            with open(pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            return pickle_data

    raise ValueError("No pickle file found in both dataset root and scene folder.")


def aggregate_frames_in_sequences(pickle_data):
    sequences_to_frames_lookup = {}
    for data in pickle_data:
        sequence_id = data['sequence_id']
        frame_id = data['frame_id']
        if sequence_id not in sequences_to_frames_lookup:
            sequences_to_frames_lookup[sequence_id] = []
        sequences_to_frames_lookup[sequence_id].append(frame_id)
    return sequences_to_frames_lookup


def build_frame_id_to_annotations_lookup(pickle_data):
    id_to_annotations_lookup = {}
    for data in pickle_data:
        frame_id = data['frame_id']
        if frame_id not in id_to_annotations_lookup:
            id_to_annotations_lookup[frame_id] = data
    return id_to_annotations_lookup


def get_frame_ids_for_scene(once: ONCE,
                            scene_id: str) -> list:
    frames_folder_path = os.path.join(once.data_folder, scene_id, 'lidar_roof')
    raw_frame_files = os.listdir(frames_folder_path)

    frame_ids = [os.path.basename(file).split('.')[0] for file in raw_frame_files]
    return sorted(frame_ids)
