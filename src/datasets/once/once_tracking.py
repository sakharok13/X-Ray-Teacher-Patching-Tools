import numpy as np
import os
import json

class Instance:
    def __init__(self, instance_id, tracking, category, scene_id, frame_ids, boxes_3d):
        self.instance_id = instance_id
        self.tracking = tracking
        self.category = category
        self.scene_id = scene_id
        self.frame_ids = []
        self.boxes_3d = []

def get_annotations_file_name(dataset_root, seq_id):
    return dataset_root + '/data/' + str(seq_id) + '/' + str(seq_id) + '.json'

def get_tracking_file_name(dataset_root, seq_id):
    return dataset_root + '/data/' + str(seq_id) + '/' + str(seq_id) + '_tracked.json'

def track_instances(dataset, dataset_root, seq_id, frame_ids):
    instances_dict = {}

    anno_json = get_annotations_file_name(dataset_root, seq_id)
    with open(anno_json, 'r') as json_file:
        data = json.load(json_file)

    i = 0
    while i < (len(frame_ids) - 1):


        current_annotations = dataset.get_frame_anno(seq_id, frame_ids[i])
        if current_annotations is None:
            i += 1
            continue

        next_i = None
        for j in range(i + 1, len(frame_ids)):
            next_annotations = dataset.get_frame_anno(seq_id, frame_ids[j])
            if next_annotations is not None:
                next_i = j
                break

        frame_id = frame_ids[i]
        print("processing frame " + str(i) + " of " +
              str(len(frame_ids)) + " id: " + str(frame_id))
        # points = dataset.load_point_cloud(seq_id, frame_id)

        current_categories = current_annotations['names']
        current_boxes_3d = current_annotations['boxes_3d']

        next_frame_id = frame_ids[next_i]
        next_annotations = dataset.get_frame_anno(seq_id, next_frame_id)

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
                # instance.clouds.append(get_instance_ptcloud(points, box))
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
                # new_instance.clouds.append(get_instance_ptcloud(points, box))
                instances_dict[len(instances_dict)] = new_instance

        data["frames"][next_i]["annos"]["instance_ids"] = next_instance_ids

        i += 1

    output_filename = get_tracking_file_name(dataset_root, seq_id)
    with open(output_filename, "w") as output_file:
        json.dump(data, output_file, indent=4)