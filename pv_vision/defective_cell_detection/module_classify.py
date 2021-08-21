import json


def count_defects(ann_path, defect_dict):
    """count the number of defects on solar module

    Parameters
    ----------
    ann_path: str or pathlib.PosixPath
    The path of the json files that contain the information of predictions.
    Format should follow supervisely prediction format.

    defect_dict: dict
    The defects to be detected. Key is the defect name in the prediction file, and value is the number of defects.
    E.g.
    {'crack_bbox_yolo': 0, 'intra_bbox_yolo': 0, 'oxygen_bbox_yolo': 0, 'solder_bbox_yolo': 0}

    Returns
    -------
    defect_dict: dict
    The counted number of defects
    """
    with open(ann_path, 'r') as f:
            data = json.load(f)

    for defect in data["objects"]:
        if defect["classTitle"] in defect_dict.keys():
            defect_dict[defect["classTitle"]] += 1

    return defect_dict
