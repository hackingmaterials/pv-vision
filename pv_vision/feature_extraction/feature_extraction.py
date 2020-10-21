import numpy as np


def extract_length(line):
    line_temp = line[1:]
    line_temp.append(line[-1])
    length = np.sqrt(np.sum((line_temp - np.array(line))**2, axis=1))
    length_tot = np.sum(length)

    return length_tot


def extract_angle(line):
    x = list(zip(*line))[0]
    y = list(zip(*line))[1]
    k, b = np.polyfit(x, y, 1)

    return k, b


def flatten_list(list_comb):
    list_flatten = []
    list_comb = list(list_comb)
    for element in list_comb:
        list_flatten += element

    return list_flatten
