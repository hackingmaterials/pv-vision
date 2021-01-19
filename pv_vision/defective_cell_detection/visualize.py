import cv2 as cv
import json


def draw_rec(ann_path, image, color, thickness=3):
    with open(ann_path, 'r') as file:
        data = json.load(file)
    if len(data['objects']) > 0:
        for obj in data['objects']:
            classTitle = obj['classTitle']
            points = obj['points']['exterior']

            cv.rectangle(image, tuple(points[0]), tuple(points[1]), color[classTitle], thickness)