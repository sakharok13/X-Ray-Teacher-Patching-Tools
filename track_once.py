import argparse
import os
import numpy as np
import multiprocessing
import pickle

from tqdm import tqdm
from functools import partial

from src.datasets.once.once_utils import aggregate_frames_in_sequences, build_frame_id_to_annotations_lookup


class Instance:
    def __init__(self, instance_id, tracking, category, scene_id, frame_ids, boxes_3d):
        self.instance_id = instance_id
        self.tracking = tracking
        self.category = category
        self.scene_id = scene_id
        self.frame_ids = []
        self.boxes_3d = []


def track_instances(scene_id: str,
                    sequences_to_frames_lookup: dict,
                    frame_id_to_annotations_lookup: dict,
                    save_dir: str,
                    force_overwrite: bool):
    instances_dict = {}
    output_file_path = os.path.normpath(os.path.join(save_dir, f"once_raw_small_{scene_id}.pkl"))

    if not force_overwrite and os.path.exists(output_file_path):
        print(f"Skipping sequence: {scene_id}")
        return

    frames = sequences_to_frames_lookup[scene_id]
    current_and_next_iterator = zip(frames, frames[1:])

    scene_annotations = []
    next_frame_data = []

    for frame_id, next_frame_id in tqdm(current_and_next_iterator, desc=f"Sequence {scene_id}", total=len(frames)):
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
                                    scene_id=scene_id,
                                    frame_ids=[],
                                    boxes_3d=[])
                instance.frame_ids.append(frame_id)
                box = current_boxes_3d[idx]
                instance.boxes_3d.append(box)
                instances_dict[len(instances_dict)] = instance

            # fill instance ids for the first frame with annotations
            current_frame_data["annos"]["instance_ids"] = first_instance_ids

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
                                        scene_id=scene_id,
                                        frame_ids=[],
                                        boxes_3d=[])

                instances_dict[len(instances_dict)] = new_instance
                new_instance.frame_ids.append(next_frame_id)
                box = next_boxes_3d[idx]
                new_instance.boxes_3d.append(box)
                instances_dict[len(instances_dict)] = new_instance

        next_frame_data["annos"]["instance_ids"] = next_instance_ids
        scene_annotations.append(current_frame_data)

    scene_annotations.append(next_frame_data)

    try:
        with open(output_file_path, 'wb') as destination_file:
            pickle.dump(scene_annotations, destination_file)
    except FileNotFoundError:
        print(f"Source file not found.")


def parallel_process(scene,
                     sequences_to_frames_lookup,
                     frame_id_to_annotations_lookup,
                     save_dir,
                     force_overwrite):
    num_workers = multiprocessing.cpu_count()
    print(f"Detected CPUs: {num_workers}")

    process_single_sequence = partial(
        track_instances,
        sequences_to_frames_lookup=sequences_to_frames_lookup,
        frame_id_to_annotations_lookup=frame_id_to_annotations_lookup,
        save_dir=save_dir,
        force_overwrite=force_overwrite
    )

    with multiprocessing.Pool(num_workers) as p:
        list(tqdm(p.imap(process_single_sequence, scene), total=len(scene)))


def parse_arguments():
    parser = argparse.ArgumentParser(description='patch scene arguments')
    parser.add_argument('--split', type=str,
                        choices=['train', 'val', 'test', 'raw_small', 'raw_medium', 'raw_large'],
                        default=None, help='Once dataset split.')
    parser.add_argument('--dataroot', type=str, default=None, help='Data root location.')
    parser.add_argument('--save_dir', type=str, default='./out', help='Directory to save tracked files.')
    parser.add_argument('--force_overwrite', action='store_true', help='Overwrite saved files.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    split = args.split
    dataset_root = args.dataroot
    save_dir_path = args.save_dir
    overwrite = args.force_overwrite

    assert dataset_root is not None, \
        "Dataset root should be specified."

    assert split is not None, \
        "Split was not specified."
    split_file = os.path.join(dataset_root, 'ImageSets', f"{split}.txt")

    os.makedirs(save_dir_path, exist_ok=True)

    scenes_path = os.path.join(dataset_root, 'data')
    scenes = set(map(lambda x: x.strip(), open(split_file).readlines()))

    path = os.path.join(dataset_root, 'once_raw_small.pkl')
    with open(path, 'rb') as file:
        pickle_data = pickle.load(file)

    sequences_to_frames_lookup = aggregate_frames_in_sequences(pickle_data)
    frame_id_to_annotations_lookup = build_frame_id_to_annotations_lookup(pickle_data)

    parallel_process(scenes,
                     sequences_to_frames_lookup,
                     frame_id_to_annotations_lookup,
                     save_dir_path,
                     overwrite)
