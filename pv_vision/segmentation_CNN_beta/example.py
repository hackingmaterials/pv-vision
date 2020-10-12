from seg_cnn import *
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import zlib, base64
from scipy import signal
import os

path = 'ann/1-4-A6CB2-53-07_28_22_05_05.png.json'
name = path.split('.')[0]
name = name.split('/')[-1]
image = cv.imread('img/'+name+'.png' )
# plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

mask = load_mask(path, image)
plt.imshow(mask, "gray")

corners = find_module_corner(mask, method=1)

wrap = perspective_transform(image, corners, 600, 300)
# plt.imshow(cv.cvtColor(wrap, cv.COLOR_BGR2RGB))

peak_x, peak_y = find_cell_corner(wrap)

crop_cell(wrap, peak_x, peak_y, saveplace="crop/", plot_bool=1)


# # Now do the segment on all the solar modules

filepaths = os.listdir(r'./ann/')

N = 0
N_err = 0
for file in filepaths:
    N += 1
    try:
        path = 'ann/'+file
        name = file.split('.')[0]
        image = cv.imread('img/'+name+'.png' )

        mask = load_mask(path, image)
        corners = find_module_corner(mask, method=0)
        wrap = perspective_transform(image, corners, 600, 300)
        peak_x, peak_y = find_cell_corner(wrap)
        if len(peak_x) == 15 and len(peak_y) == 7:
            os.mkdir("crop/"+name)
            crop_cell(wrap, peak_x, peak_y, saveplace="crop/"+name+"/")
        else:
            print(name + " trying method2")
            corners = find_module_corner(mask, method=1)
            wrap = perspective_transform(image, corners, 600, 300)
            peak_x, peak_y = find_cell_corner(wrap)
            if len(peak_x) == 15 and len(peak_y) == 7:
                os.mkdir("crop/"+name)
                crop_cell(wrap, peak_x, peak_y, saveplace="crop/"+name+"/")
            else:
                N_err+=1
                print(name+" seg error")      
    except:
        print(name+" run error")
        N_err += 1

print("total images: "+str(N))
print("failed images:"+ str(N_err))
print("accuracy: " + str(1-N_err/N))
