import json

def count_defects(path):
    with open(path, 'r') as f:
            data = json.load(f)
            
    crack = 0
    oxygen = 0
    intra = 0
    solder = 0

    for defect in data["objects"]:
        if defect["classTitle"] == "crack_bbox_yolo":
            crack += 1
        elif defect["classTitle"] == "oxygen_bbox_yolo":
            oxygen += 1
        elif defect["classTitle"] == "intra_bbox_yolo":
            intra += 1
        elif defect["classTitle"] == "solder_bbox_yolo":
            solder += 1

    return crack, oxygen, intra, solder
    