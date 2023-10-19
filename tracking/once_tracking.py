import numpy as np
import functools
import os
import os.path as osp
from collections import defaultdict
import cv2
import json
import multiprocessing


def get_annotations_file_name(datafolder_root, seq_id):
    return datafolder_root + '/data/' + str(seq_id) + '/' + str(seq_id) + '.json'

def split_info_loader_helper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        split_file_path = func(*args, **kwargs)
        if not osp.isfile(split_file_path):
            split_list = []
        else:
            split_list = set(
                map(lambda x: x.strip(), open(split_file_path).readlines()))
        return split_list

    return wrapper

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
    camera_names = [
        'cam01',
        'cam03',
        'cam05',
        'cam06',
        'cam07',
        'cam08',
        'cam09']
    camera_tags = [
        'top',
        'top2',
        'left_back',
        'left_front',
        'right_front',
        'right_back',
        'back']

    def __init__(self, dataset_root, typeds):
        self.dataset_root = dataset_root
        self.data_folder = osp.join(self.dataset_root, 'data')
        self._collect_basic_infos(typeds)

    @property
    @split_info_loader_helper
    def train_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'train.txt')

    @property
    @split_info_loader_helper
    def val_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'val.txt')

    @property
    @split_info_loader_helper
    def test_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'test.txt')

    @property
    @split_info_loader_helper
    def raw_small_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'raw_small.txt')

    @property
    @split_info_loader_helper
    def raw_medium_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'raw_medium.txt')

    @property
    @split_info_loader_helper
    def raw_large_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'raw_large.txt')

    def _find_split_name(self, seq_id):
        if seq_id in self.raw_small_split_list:
            return 'raw_small'
        elif seq_id in self.raw_medium_split_list:
            return 'raw_medium'
        elif seq_id in self.raw_large_split_list:
            return 'raw_large'
        if seq_id in self.train_split_list:
            return 'train'
        if seq_id in self.test_split_list:
            return 'test'
        if seq_id in self.val_split_list:
            return 'val'
        print("sequence id {} corresponding to no split".format(seq_id))
        raise NotImplementedError

    def _collect_basic_infos(self, typeds):
        self.train_info = defaultdict(dict)
        self.val_info = defaultdict(dict)
        self.test_info = defaultdict(dict)
        self.raw_small_info = defaultdict(dict)
        self.raw_medium_info = defaultdict(dict)
        self.raw_large_info = defaultdict(dict)

        attr_list = [typeds]

        for attr in attr_list:
            if getattr(self, '{}_split_list'.format(attr)) is not None:
                split_list = getattr(self, '{}_split_list'.format(attr))
                split_list = {'000092'}
                info_dict = getattr(self, '{}_info'.format(attr))
                for seq in split_list:
                    anno_file_path = osp.join(
                        self.data_folder, seq, '{}.json'.format(seq))
                    if not osp.isfile(anno_file_path):
                        print("no annotation file for sequence {}".format(seq))
                        raise FileNotFoundError
                    anno_file = json.load(open(anno_file_path, 'r'))
                    frame_list = list()
                    for frame_anno in anno_file['frames']:
                        frame_list.append(str(frame_anno['frame_id']))
                        info_dict[seq][frame_anno['frame_id']] = {
                            'pose': frame_anno['pose'],
                        }
                        info_dict[seq][frame_anno['frame_id']
                                       ]['calib'] = dict()
                        for cam_name in self.__class__.camera_names:
                            info_dict[seq][frame_anno['frame_id']]['calib'][cam_name] = {
                                'cam_to_velo': np.array(anno_file['calib'][cam_name]['cam_to_velo']),
                                'cam_intrinsic': np.array(anno_file['calib'][cam_name]['cam_intrinsic']),
                                'distortion': np.array(anno_file['calib'][cam_name]['distortion'])
                            }
                        if 'annos' in frame_anno.keys():
                            info_dict[seq][frame_anno['frame_id']
                                           ]['annos'] = frame_anno['annos']
                    info_dict[seq]['frame_list'] = sorted(frame_list)

    def _collect_basic_infos_tracked(self):
        self.train_info_tracked = defaultdict(dict)
        self.val_info_tracked = defaultdict(dict)
        self.test_info_tracked = defaultdict(dict)
        self.raw_small_info_tracked = defaultdict(dict)
        self.raw_medium_info_tracked = defaultdict(dict)
        self.raw_large_info_tracked = defaultdict(dict)

        for attr in [
            'train',
            'val',
            'test',
            'raw_small',
            'raw_medium',
                'raw_large']:
            if getattr(self, '{}_split_list'.format(attr)) is not None:
                split_list = getattr(self, '{}_split_list'.format(attr))
                info_dict = getattr(self, '{}_info_tracked'.format(attr))
                for seq in split_list:
                    anno_file_path = osp.join(
                        self.data_folder, seq, '{}_tracked.json'.format(seq))
                    if not osp.isfile(anno_file_path):
                        print("no annotation file for sequence {}".format(seq))
                        raise FileNotFoundError
                    anno_file = json.load(open(anno_file_path, 'r'))
                    frame_list = list()
                    for frame_anno in anno_file['frames']:
                        frame_list.append(str(frame_anno['frame_id']))
                        info_dict[seq][frame_anno['frame_id']] = {
                            'pose': frame_anno['pose'],
                        }
                        info_dict[seq][frame_anno['frame_id']
                                       ]['calib'] = dict()
                        for cam_name in self.__class__.camera_names:
                            info_dict[seq][frame_anno['frame_id']]['calib'][cam_name] = {
                                'cam_to_velo': np.array(anno_file['calib'][cam_name]['cam_to_velo']),
                                'cam_intrinsic': np.array(anno_file['calib'][cam_name]['cam_intrinsic']),
                                'distortion': np.array(anno_file['calib'][cam_name]['distortion'])
                            }
                        if 'annos' in frame_anno.keys():
                            info_dict[seq][frame_anno['frame_id']
                                           ]['annos'] = frame_anno['annos']
                    info_dict[seq]['frame_list'] = sorted(frame_list)

    def get_frame_anno(self, seq_id, frame_id):
        split_name = self._find_split_name(seq_id)
        frame_info = getattr(self, '{}_info'.format(split_name))[
            seq_id][frame_id]
        if 'annos' in frame_info:
            return frame_info['annos']
        return None

    def load_point_cloud(self, seq_id, frame_id):
        bin_path = osp.join(
            self.data_folder,
            seq_id,
            'lidar_roof',
            '{}.bin'.format(frame_id))
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points

    def load_image(self, seq_id, frame_id, cam_name):
        cam_path = osp.join(
            self.data_folder,
            seq_id,
            cam_name,
            '{}.jpg'.format(frame_id))
        img_buf = cv2.cvtColor(cv2.imread(cam_path), cv2.COLOR_BGR2RGB)
        return img_buf

    def undistort_image(self, seq_id, frame_id):
        img_list = []
        split_name = self._find_split_name(seq_id)
        frame_info = getattr(self, '{}_info'.format(split_name))[
            seq_id][frame_id]
        for cam_name in self.__class__.camera_names:
            img_buf = self.load_image(seq_id, frame_id, cam_name)
            cam_calib = frame_info['calib'][cam_name]
            h, w = img_buf.shape[:2]
            cv2.getOptimalNewCameraMatrix(cam_calib['cam_intrinsic'],
                                          cam_calib['distortion'],
                                          (w, h), alpha=0.0, newImgSize=(w, h))
            img_list.append(
                cv2.undistort(
                    img_buf,
                    cam_calib['cam_intrinsic'],
                    cam_calib['distortion'],
                    newCameraMatrix=cam_calib['cam_intrinsic']))
        return img_list

    def undistort_image_v2(self, seq_id, frame_id):
        img_list = []
        new_cam_intrinsic_dict = dict()
        split_name = self._find_split_name(seq_id)
        frame_info = getattr(self, '{}_info'.format(split_name))[
            seq_id][frame_id]
        for cam_name in self.__class__.camera_names:
            img_buf = self.load_image(seq_id, frame_id, cam_name)
            cam_calib = frame_info['calib'][cam_name]
            h, w = img_buf.shape[:2]
            new_cam_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
                cam_calib['cam_intrinsic'], cam_calib['distortion'], (w, h), alpha=0.0, newImgSize=(
                    w, h))
            img_list.append(cv2.undistort(img_buf, cam_calib['cam_intrinsic'],
                                          cam_calib['distortion'],
                                          newCameraMatrix=new_cam_intrinsic))
            new_cam_intrinsic_dict[cam_name] = new_cam_intrinsic
        return img_list, new_cam_intrinsic_dict

    def project_lidar_to_image(self, seq_id, frame_id):
        points = self.load_point_cloud(seq_id, frame_id)

        split_name = self._find_split_name(seq_id)
        frame_info = getattr(self, '{}_info'.format(split_name))[
            seq_id][frame_id]
        points_img_dict = dict()
        img_list, new_cam_intrinsic_dict = self.undistort_image_v2(
            seq_id, frame_id)
        for cam_no, cam_name in enumerate(self.__class__.camera_names):
            calib_info = frame_info['calib'][cam_name]
            cam_2_velo = calib_info['cam_to_velo']
            cam_intri = np.hstack(
                [new_cam_intrinsic_dict[cam_name], np.zeros((3, 1), dtype=np.float32)])
            point_xyz = points[:, :3]
            points_homo = np.hstack([point_xyz, np.ones(
                point_xyz.shape[0], dtype=np.float32).reshape((-1, 1))])
            points_lidar = np.dot(points_homo, np.linalg.inv(cam_2_velo).T)
            mask = points_lidar[:, 2] > 0
            points_lidar = points_lidar[mask]
            points_img = np.dot(points_lidar, cam_intri.T)
            points_img = points_img / points_img[:, [2]]
            img_buf = img_list[cam_no]
            for point in points_img:
                try:
                    cv2.circle(
                        img_buf, (int(
                            point[0]), int(
                            point[1])), 2, color=(
                            0, 0, 255), thickness=-1)
                except BaseException:
                    print(int(point[0]), int(point[1]))
            points_img_dict[cam_name] = img_buf
        return points_img_dict

    @staticmethod
    def rotate_z(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    def project_boxes_to_image(self, seq_id, frame_id):
        split_name = self._find_split_name(seq_id)
        if split_name not in ['train', 'val']:
            print("seq id {} not in train/val, has no 2d annotations".format(seq_id))
            return
        frame_info = getattr(self, '{}_info'.format(split_name))[
            seq_id][frame_id]
        img_dict = dict()
        img_list, new_cam_intrinsic_dict = self.undistort_image_v2(
            seq_id, frame_id)
        for cam_no, cam_name in enumerate(self.__class__.camera_names):
            img_buf = img_list[cam_no]

            calib_info = frame_info['calib'][cam_name]
            cam_2_velo = calib_info['cam_to_velo']
            cam_intri = np.hstack(
                [new_cam_intrinsic_dict[cam_name], np.zeros((3, 1), dtype=np.float32)])

            cam_annos_3d = np.array(frame_info['annos']['boxes_3d'])

            corners_norm = np.stack(np.unravel_index(np.arange(8), [2, 2, 2]), axis=1).astype(
                np.float32)[[0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 5, 1, 3, 7, 6, 2], :] - 0.5
            corners = np.multiply(
                cam_annos_3d[:, 3: 6].reshape(-1, 1, 3), corners_norm)
            rot_matrix = np.stack(
                list([np.transpose(self.rotate_z(box[-1])) for box in cam_annos_3d]), axis=0)
            corners = np.einsum('nij,njk->nik', corners, rot_matrix) + \
                cam_annos_3d[:, :3].reshape((-1, 1, 3))

            for i, corner in enumerate(corners):
                points_homo = np.hstack(
                    [corner, np.ones(corner.shape[0], dtype=np.float32).reshape((-1, 1))])
                points_lidar = np.dot(points_homo, np.linalg.inv(cam_2_velo).T)
                mask = points_lidar[:, 2] > 0
                points_lidar = points_lidar[mask]
                points_img = np.dot(points_lidar, cam_intri.T)
                points_img = points_img / points_img[:, [2]]
                if points_img.shape[0] != 16:
                    continue
                for j in range(15):
                    cv2.line(img_buf, (int(points_img[j][0]), int(points_img[j][1])), (int(
                        points_img[j + 1][0]), int(points_img[j + 1][1])), (0, 255, 0), 2, cv2.LINE_AA)

            cam_annos_2d = frame_info['annos']['boxes_2d'][cam_name]

            for box2d in cam_annos_2d:
                box2d = list(map(int, box2d))
                if box2d[0] < 0:
                    continue
                cv2.rectangle(img_buf, tuple(box2d[:2]), tuple(
                    box2d[2:]), (255, 0, 0), 2)

            img_dict[cam_name] = img_buf
        return img_dict


class Instance:
    def __init__(self, instance_id, tracking, category, scene_id, frame_ids, boxes_3d):
        self.instance_id = instance_id
        self.tracking = tracking
        self.category = category
        self.scene_id = scene_id
        self.frame_ids = []
        self.boxes_3d = []

def get_tracking_file_name(dataset_root, seq_id):
    return dataset_root + '/data/' + str(seq_id) + '/' + str(seq_id) + '_tracked.json'

def track_instances(dataset, dataset_root, seq_id):
    instances_dict = {}

    anno_json = get_annotations_file_name(dataset_root, seq_id)
    with open(anno_json, 'r') as json_file:
        data = json.load(json_file)

    len_frames = len(data['frames'])

    i = 0
    while i < (len_frames - 1):

        current_annotations = data['frames'][i].get('annos', None)  # dataset.get_frame_anno(seq_id, frame_ids[i])
        if current_annotations is None:
            i += 1
            continue

        next_i = None
        for j in range(i + 1, len_frames):
            next_annotations = data['frames'][j].get('annos', None)  # dataset.get_frame_anno(seq_id, frame_ids[j])
            if next_annotations is not None:
                next_i = j
                break

        if next_i == None:
            break

        print("processing frame " + str(i+1) + " of " +
              str(len_frames) )

        frame_id = data['frames'][i].get('frame_id', None)
        next_frame_id = data['frames'][next_i].get('frame_id', None)

        current_categories = current_annotations['names']
        current_boxes_3d = current_annotations['boxes_3d']

        # next_frame_id = frame_ids[next_i]
        next_annotations = data['frames'][next_i].get('annos', None)  # dataset.get_frame_anno(seq_id, next_frame_id)

        next_categories = next_annotations['names']
        next_boxes_3d = next_annotations['boxes_3d']

        if not instances_dict:
            # first frame
            first_instance_ids = []

            for idx, category in enumerate(current_categories):
                first_instance_ids.append(len(instances_dict))
                instance = Instance(instance_id=len(instances_dict),
                                    tracking=[],
                                    category=category,
                                    scene_id=seq_id,
                                    frame_ids=[],
                                    boxes_3d=[])
                instance.frame_ids.append(frame_id)
                box = current_boxes_3d[idx]
                instance.boxes_3d.append(box)
                instances_dict[len(instances_dict)] = instance

            # fill instance ids for the first frame with annotations
            data["frames"][i]["annos"]["instance_ids"] = first_instance_ids

        next_instance_ids = []

        for idx, next_category in enumerate(next_categories):
            next_box_3d = next_boxes_3d[idx]
            next_center = [next_box_3d[0], next_box_3d[1], next_box_3d[2]]
            matched_instance_id = None
            min_distance = max(next_box_3d[3:6]) * 2

            for instance_id, instance in instances_dict.items():
                if frame_id in instance.frame_ids and instance.category == next_category:
                    current_box_3d = instance.boxes_3d[-1]
                    current_center = [
                        current_box_3d[0],
                        current_box_3d[1],
                        current_box_3d[2]]
                    distance = np.linalg.norm(
                        np.array(current_center) - np.array(next_center))

                    if distance < min_distance:
                        min_distance = distance
                        matched_instance_id = instance_id

            if matched_instance_id is not None:
                # matched
                next_instance_ids.append(matched_instance_id)
                instances_dict[matched_instance_id].frame_ids.append(
                    next_frame_id)
                instances_dict[matched_instance_id].boxes_3d.append(
                    next_box_3d)

            else:
                # unmatched object
                next_instance_ids.append(len(instances_dict))
                new_instance = Instance(instance_id=len(instances_dict),
                                        tracking=[],
                                        category=next_category,
                                        scene_id=seq_id,
                                        frame_ids=[],
                                        boxes_3d=[])

                instances_dict[len(instances_dict)] = new_instance
                new_instance.frame_ids.append(next_frame_id)
                box = next_boxes_3d[idx]
                new_instance.boxes_3d.append(box)
                instances_dict[len(instances_dict)] = new_instance

        data["frames"][next_i]["annos"]["instance_ids"] = next_instance_ids

        i += 1

    output_filename = get_tracking_file_name(dataset_root, seq_id)
    with open(output_filename, "w") as output_file:
        json.dump(data, output_file)

def process_batch(scenes_batch, dataset, dataset_root):
    for seq_id in scenes_batch:
        track_instances(dataset, dataset_root, seq_id)

def parallel_process(scenes, dataset, dataset_root):
    num_workers = multiprocessing.cpu_count()
    print("cpu count: " + str(multiprocessing.cpu_count()))
    scenes_count = len(scenes)

    batches = [scenes[i:i + len(scenes) // num_workers] for i in range(0, len(scenes), len(scenes) // num_workers)]


    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     executor.map(process_batch, batches, [dataset] * num_workers, [dataset_root] * num_workers)

    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     pool.starmap(track_instances, [(dataset, dataset_root, seq_id) for seq_id in scenes])


if __name__ == '__main__':
    dataset_root = "./temp/ONCE"
    dataset = ONCE(dataset_root, 'train')
    scenes_path = osp.join(dataset_root, 'data')
    scenes = os.listdir(scenes_path)

    # parallel_process(scenes, dataset, dataset_root)
    track_instances(dataset, dataset_root, scenes[0]) # for one scene!!! 
