from shapely.geometry import Polygon
import numpy as np
import math



def project_box_to_corners2d(box):
    cx, cy, cz, l, w, h, theta = box

    hl = l / 2
    hw = w / 2

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    corners_2d = [
        (cx - hl * cos_theta - hw * sin_theta, cy - hl * sin_theta + hw * cos_theta),
        (cx + hl * cos_theta - hw * sin_theta, cy + hl * sin_theta + hw * cos_theta),
        (cx + hl * cos_theta + hw * sin_theta, cy + hl * sin_theta - hw * cos_theta),
        (cx - hl * cos_theta + hw * sin_theta, cy - hl * sin_theta - hw * cos_theta)
    ]

    return corners_2d
def intersection_area(rect1, rect2):
    polygon1 = Polygon(rect1)
    polygon2 = Polygon(rect2)
    intersection = polygon1.intersection(polygon2)

    if intersection.is_empty:
        return 0.0

    return intersection.area

def calculate_iou(box1, box2):
    """Calculates IOU for two ONCE-format boxes
    :param box1: cx, cy, cz, l, w, h, theta
    :param box2: cx, cy, cz, l, w, h, theta
    """
    cx1, cy1, cz1, l1, w1, h1, theta1 = box1
    cx2, cy2, cz2, l2, w2, h2, theta2 = box2

    rect1 = project_box_to_corners2d(box1)
    rect2 = project_box_to_corners2d(box2)

    int_area = intersection_area(rect1, rect2)

    h = (h1+h2)/2

    intersection = int_area*h

    volume1 = l1 * w1 * h1
    volume2 = l2 * w2 * h2

    union = volume1 + volume2 - intersection
    iou = intersection / union

    return iou

# Example usage:
box1 = [-3.460779890336198, 11.889099506707907, -0.8206982783150591, 4.555971205891034, 1.8030093908309937, 1.5509192471702893, 5.2348885297696235]# [0,0,0,2,2,2,0]
box2 = [-3.460779890336198, 11.889099506707907, -0.8206982783150591, 4.555971205891034, 1.8030093908309937, 1.5509192471702893, 3.2348885297696235]# [1,1,1,2,2,2,0]


iou = calculate_iou(box1, box2)
print("IoU:", iou)