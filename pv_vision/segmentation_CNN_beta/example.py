from seg_cnn import *
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import zlib, base64
from scipy import signal
import os

filepaths = os.listdir('img/')
if '.DS_Store' in filepaths:
    filepaths.remove('.DS_Store')

N = 0
N_err = 0
for file in filepaths:
    N += 1
    try:
        name = file.split('.')[0]
        path = 'ann/'+name+'.png.json'
        image = cv.imread('img/'+name+'.png')

        mask, mask_center = load_mask(path, image)
        corners = find_module_corner(mask, mask_center, method=0, displace=3)
        wrap = perspective_transform(image, corners, 600, 300)
        peak_x, peak_y = find_cell_corner(wrap)
        if len(peak_x) > 12 and len(peak_y) > 5:
            cv.imwrite("perspective_transform/"+name+".png", wrap)
        else:
            print(name + " trying method2")
            corners = find_module_corner(mask, mask_center, method=1, displace=3)
            wrap = perspective_transform(image, corners, 600, 300)
            peak_x, peak_y = find_cell_corner(wrap)
            if len(peak_x) > 12 and len(peak_y) > 5:
                cv.imwrite("perspective_transform/"+name+".png", wrap)
            else:
                corners = find_module_corner(mask, mask_center, method=0,
                                             displace=3, corner_center=True, center_displace=50)
                wrap = perspective_transform(image, corners, 600, 300)
                peak_x, peak_y = find_cell_corner(wrap)
                if len(peak_x) > 12 and len(peak_y) > 5:
                    cv.imwrite("perspective_transform/"+name+".png", wrap)
                else:
                    corners = find_module_corner(mask, mask_center, method=1,
                                                 displace=3, corner_center=True, center_displace=50)
                    wrap = perspective_transform(image, corners, 600, 300)
                    peak_x, peak_y = find_cell_corner(wrap)
                    if len(peak_x) > 12 and len(peak_y) > 5:
                        cv.imwrite("perspective_transform/"+name+".png", wrap)
                    else:
                        N_err += 1
                        print(name+" seg error")
                        #with open("error.csv",'a') as f:
                        #    f.write(name+'.png\n')
                        #shutil.copyfile('img/'+name+'.png', 'error_all/'+name+'.png')   
    except:
        print(name+" run error")
        N_err += 1
        #with open("error2.csv",'a') as f:
        #    f.write(name+'.png\n')
        #shutil.copyfile('img/'+name+'.png', 'error_all/'+name+'.png')

print("total images: "+str(N))
print("failed images:" + str(N_err))
print("accuracy: " + str(1 - N_err / N))
