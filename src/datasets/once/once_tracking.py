import numpy as np
import functools
import os
import os.path as osp
from collections import defaultdict
import cv2
import json
import multiprocessing
import pickle
from tqdm import tqdm
from functools import partial
from once_utils import ONCE


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
        assert frame_id not in id_to_annotations_lookup
        id_to_annotations_lookup[frame_id] = data
    return id_to_annotations_lookup


def track_instances(seq_id, dataset, dataset_root, pickle_data, sequences_to_frames_lookup,
                    frame_id_to_annotations_lookup, outfile):
    instances_dict = {}
    outpath = osp.join(dataset_root, outfile + '_' + str(seq_id) + '.pkl')

    if os.path.exists(outpath):
        print(f"Skipping sequence: {seq_id}")
        return

    frames = sequences_to_frames_lookup[seq_id]
    current_and_next_iterator = zip(frames, frames[1:])

    scene_annotations = []

    for frame_id, next_frame_id in tqdm(current_and_next_iterator, desc=f"Sequence {seq_id}", total=len(frames) - 1):
        current_frame_data = frame_id_to_annotations_lookup[frame_id]
        current_annotations = current_frame_data['annos']

        current_categories = current_annotations['name']
        current_boxes_3d = current_annotations['boxes_3d']

        next_frame_data = frame_id_to_annotations_lookup[next_frame_id]
        next_annotations = next_frame_data.get('annos', None)

        next_categories = next_annotations['name']
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
            current_frame_data["annos"]["instance_ids"] = first_instance_ids
            scene_annotations.append(current_frame_data)

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

        next_frame_data["annos"]["instance_ids"] = next_instance_ids
        scene_annotations.append(current_frame_data)

    try:
        with open(outpath, 'wb') as destination_file:
            pickle.dump(scene_annotations, destination_file)
    except FileNotFoundError:
        print(f"Source file not found.")


def parallel_process(dataset,
                     dataset_root,
                     scene,
                     pickle_data,
                     sequences_to_frames_lookup,
                     frame_id_to_annotations_lookup,
                     outfile):
    num_workers = 2  # multiprocessing.cpu_count()
    print(f"Detected CPUs: {num_workers}")

    process_single_sequence = partial(
        track_instances,
        dataset=dataset,
        dataset_root=dataset_root,
        pickle_data=pickle_data,
        sequences_to_frames_lookup=sequences_to_frames_lookup,
        frame_id_to_annotations_lookup=frame_id_to_annotations_lookup,
        outfile=outfile,
    )

    with multiprocessing.Pool(num_workers) as p:
        list(tqdm(p.imap(process_single_sequence, scene), total=len(scene)))


if __name__ == '__main__':
    dataset_root = "D:/"
    dataset = ONCE(dataset_root, 'raw_small')
    scenes_path = osp.join(dataset_root, 'data')
    scenes = sorted(os.listdir(scenes_path))

    path = osp.join(dataset_root, 'once_raw_small.pkl')
    outpath = osp.join(dataset_root, 'once_raw_small_track.pkl')
    with open(path, 'rb') as file:
        pickle_data = pickle.load(file)

    sequences_to_frames_lookup = aggregate_frames_in_sequences(pickle_data)
    frame_id_to_annotations_lookup = build_frame_id_to_annotations_lookup(pickle_data)
    outfile = 'once_raw_small_track'

    # track_instances(scenes[0], dataset, dataset_root, pickle_data, sequences_to_frames_lookup, frame_id_to_annotations_lookup, outfile)

    parallel_process(dataset, dataset_root, scenes, pickle_data, sequences_to_frames_lookup,
                     frame_id_to_annotations_lookup, outfile)