import json

def count_defects(path, defect_name):
    with open(path, 'r') as f:
            data = json.load(f)
            
    crack = 0
    oxygen = 0
    intra = 0
    solder = 0

    for defect in data["objects"]:
        if defect["classTitle"] == defect_name['crack']:
            crack += 1
        elif defect["classTitle"] == defect_name['oxygen']:
            oxygen += 1
        elif defect["classTitle"] == defect_name['intra']:
            intra += 1
        elif defect["classTitle"] == defect_name['solder']:
            solder += 1

    return crack, oxygen, intra, solder
    