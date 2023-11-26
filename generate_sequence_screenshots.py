import os
import sys
from typing import Optional

import numpy as np
import open3d as o3d
from pyquaternion import Quaternion


def __generate_screenshots(points: np.ndarray,
                           bboxes: list = [],
                           camera_position: Optional[str] = None,
                           point_size: float = 2,
                           screenshot_file_name: str = 'Awesome visualisation'):
    n = points.shape[0]

    vector3d = o3d.utility.Vector3dVector(points[:, 0:3])
    o3d_pc = o3d.geometry.PointCloud(points=vector3d)
    o3d_pc.colors = o3d.utility.Vector3dVector(np.zeros((n, 3)))

    o3d_boxes = []

    for bbox in bboxes:
        x, y, z, dx, dy, dz, heading = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6]
        center = (x, y, z)
        r = Quaternion(angle=heading, axis=[0, 0, 1]).rotation_matrix
        size = np.array([dx, dy, dz])
        oriented_box = o3d.geometry.OrientedBoundingBox(center, r, size)
        oriented_box.color = [255.0 / 255.0, 215.0 / 255.0, 0.0 / 255.0]
        o3d_boxes.append(oriented_box)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(visible=False)

    render_option = vis.get_render_option()
    render_option.point_size = point_size

    vis.add_geometry(o3d_pc)
    for box in o3d_boxes:
        vis.add_geometry(box)

    if camera_position is not None:
        control = vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters(camera_position)
        control.convert_from_pinhole_camera_parameters(parameters)
        vis.update_renderer()

    vis.poll_events()
    vis.update_renderer()

    image_path = f"./{screenshot_file_name}.jpg"
    vis.capture_screen_image(image_path)
    print(f"Screenshot is saved to: {image_path}")
    vis.destroy_window()


def __get_bboxes(frame_id: str,
                 scene_descriptor: list[dict],
                 score_filtering: Optional = 0.17,
                 mode: str = 'nuscenes'):
    for frame_descriptor in scene_descriptor:
        if frame_descriptor['frame_id'] != frame_id:
            continue

        if mode.lower() == 'nuscenes':
            bboxes = frame_descriptor['boxes_lidar']
        else:
            raise Exception(f"Unknown mode {mode}")

        if score_filtering is not None:
            score_mask = frame_descriptor['score'] > score_filtering
            return bboxes[score_mask]
        else:
            return bboxes


def main():
    scene = sys.argv[1]
    gt_path = sys.argv[2]
    camera_position = sys.argv[3]
    scene_descriptor = np.load(gt_path, allow_pickle=True)

    for frame in sorted(os.listdir(scene)):
        if not frame.endswith(".bin"):
            continue

        frame_id = os.path.basename(frame).replace(".bin", "")
        print("Processing frame:", frame_id)

        gt_boxes = __get_bboxes(frame_id=frame_id,
                                scene_descriptor=scene_descriptor)

        lidar_point_clouds = np.fromfile(os.path.join(scene, frame), dtype=np.float32)
        lidar_point_clouds = lidar_point_clouds.reshape((-1, 5))[:, :4]
        __generate_screenshots(points=lidar_point_clouds,
                               bboxes=gt_boxes,
                               camera_position=camera_position,
                               point_size=2.5,
                               screenshot_file_name=frame_id)


if __name__ == '__main__':
    main()
