import cv2 as cv
import json


def draw_bbox(ann_path, image, color, thickness=3):
    """Draw bounding box of defective cells on the solar module images

    Parameters
    ----------
    ann_path: str or pathlib.PosixPath
    The path of the prediction json files from supervisely

    image: array
    The image of a solar module that is perspectively transformed

    color: dict
    The color of bbox.
    E.g.
    {"crack_bbox_yolo": [60, 124, 90],
     "solder_bbox_yolo": [190, 112, 78],
     "oxygen_bbox_yolo": [40, 64, 183],
     "intra_bbox_yolo": [103, 52, 154]}

     thickness: int
     The thickness of the bbox. Default is 3

     Returns
     -------
     function directly operates on the input image
    """
    with open(ann_path, 'r') as file:
        data = json.load(file)
    if len(data['objects']) > 0:
        for obj in data['objects']:
            classTitle = obj['classTitle']
            points = obj['points']['exterior']

            cv.rectangle(image, tuple(points[0]), tuple(points[1]), color[classTitle], thickness)