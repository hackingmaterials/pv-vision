import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


class SolarImage:
    def __init__(self, image, column=10, row=6, busbar=3):
        self.image = image
        self.resize = cv.resize(self.image, (4000, 2500))
        self.column = column
        self.row = row
        self.busbar = busbar
        
    def image_threshold(self, blur=5, blocksize=11):
        
        # image_eq = cv.equalizeHist(image_resize)
        image_blur = cv.medianBlur(self.image, blur)
        image_thre = cv.adaptiveThreshold(image_blur, 255, 
                                          cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv.THRESH_BINARY_INV, blocksize, 2)
        image_thre = cv.medianBlur(image_thre, 1)
        image_thre = cv.resize(image_thre, (4000, 2500))

        return image_thre

    def linear(self, x, a, b):
        return a * x + b

    def detectoutliers(self, data, rate=1.5, option=None):
    
        bottom = np.percentile(data, 25)
        up = np.percentile(data, 75)
        IQR = up - bottom
        outlier_step = rate * IQR

        outlier_list = ((data < bottom - outlier_step) | 
                        (data > up + outlier_step))
        if option == 0:
            for inx, flag in enumerate(outlier_list):
                if flag:
                    left = inx
                    right = inx
                    while outlier_list[left]:
                        left -= 1
                        if left <= 0:
                            left = inx
                            while outlier_list[left]:
                                left += 1
                                
                    while outlier_list[right]:
                        right += 1
                        if right >= len(outlier_list)-1:
                            right = inx
                            while outlier_list[right]:
                                right -= 1            
                    outlier_list[inx] = False
                    data[inx] = (data[left] + data[right]) / 2
            
            return data

        if option == 1:
            outlier_list = np.insert(outlier_list, 4, False)
            for i in range(5, len(outlier_list)-5):
                if outlier_list[i]:
                    outlier_list[i] = False
            
            return (1-outlier_list).astype(np.bool)

    def linear_regression(self, inxes, lines, threshold=0.3):
        ab_s = []

        for line in lines:
            # line=line_zip[8]
            ab, _ = curve_fit(self.linear, inxes, line)
            line_fit = ab[0]*inxes+ab[1]
            mse = np.mean((line-line_fit)**2)
            cookDistances = []
            outlier_inx = []
            for inx, element in enumerate(line):
                inx_i = np.delete(inxes, inx)
                line_i = np.delete(line, inx)
                ab_i, _ = curve_fit(self.linear, inx_i, line_i)
                line_fit_i = ab_i[0]*inxes+ab_i[1]
                CookDistance = (np.sum((line_fit - line_fit_i) ** 2)) / mse
                cookDistances.append(CookDistance)
                if CookDistance > threshold:
                    outlier_inx.append(inx)
            inxes_new = np.delete(inxes, outlier_inx)
            line_new = np.delete(line, outlier_inx)
            ab_new, _ = curve_fit(self.linear, inxes_new, line_new)
            ab_s.append(ab_new)
            # print(outlier_inx)
        ab_s = np.array(ab_s)

        return ab_s

    def detect_vertical_lines(self, image_thre, split=100):
        image_yarray = np.vsplit(image_thre, split)  # split the y axis
        edge_x = []
        inx_y = []
        for inx, im_array in enumerate(image_yarray):
            sum = np.sum(im_array, axis=0)
            sum_norm = sum / np.max(sum)
            mask = sum_norm < 0.95
            sum_norm[mask] = 0

            peak, _ = find_peaks(sum_norm)
            if len(peak) > 10:
                peak_new = []
                peak_new.append(peak[0])
                for i in range(1, len(peak)-1):
                    if np.abs(peak[i]-peak[i+1]) < 15:
                        peak_mean = (peak[i]+peak[i+1])/2
                        peak_new.append(peak_mean)
                    elif np.abs(peak[i]-peak[i-1]) > 15:
                        peak_new.append(peak[i])

                peak_new.append(peak[-1])
                peak_new = np.array(peak_new)
                peak_new_a = np.delete(peak_new, 0)
                peak_new_b = np.delete(peak_new, -1)
                peak_new_detect = peak_new[self.detectoutliers(
                                  np.abs(peak_new_a-peak_new_b), option=1)]

                if len(peak_new_detect) == 1 + self.column:
                    edge_x.append(peak_new_detect)
                    inx_mean = ((2 * inx + 1) * (image_thre.shape[0] / 
                                split) - 1) / 2
                    inx_y.append(inx_mean)
        edge_x = np.array(edge_x)

        vlines = list(zip(*edge_x))  # line parallel to y axis
        vlines = np.array(vlines)
        inx_y = np.array(inx_y)
        # for lines in vlines:
        #    lines_new = self.detectoutliers(lines, option=0)
        #    while np.std(lines_new) > 15:
        #        lines_new = self.detectoutliers(lines, rate=1, option=0)

        # v_abs = [] 
        # for verticaline in vlines:
        #    ab, _ = curve_fit(self.linear, inx_y, verticaline) # x = ay + b
        #    v_abs.append(ab)
        v_abs = self.linear_regression(inx_y, vlines)
        # temp1 = v_abs.copy()
        temp1 = np.delete(v_abs, -1, 0)
        # temp2 = v_abs.copy()
        temp2 = np.delete(v_abs, 0, 0)

        vline_abs = np.array(list(zip(temp1, temp2)))
        # vline_abs = [(v_abs[i],v_abs[i+1]) for i in range(0, len(v_abs), 2)] # put edges of each cell into a tuple

        return vline_abs

    def detect_horizon_lines(self, image_thre, split=100):
        image_xarray = np.hsplit(image_thre, split)  # split x axis

        edge_y = []
        inx_x = []
        for inx, im_array in enumerate(image_xarray):
            sum = np.sum(im_array, axis=1)
            sum_norm = sum / np.max(sum)
            mask = sum_norm < 0.95
            sum_norm[mask] = 0

            peak, _ = find_peaks(sum_norm)
            if len(peak) >= 19:
                peak_new = []
                peak_new.append(peak[0])
                for i in range(1, len(peak)-1):
                    if np.abs(peak[i]-peak[i+1]) < 15:
                        peak_mean = (peak[i] + peak[i + 1]) / 2
                        peak_new.append(peak_mean)
                    elif np.abs(peak[i] - peak[i - 1]) > 15:
                        peak_new.append(peak[i])

                peak_new.append(peak[-1])
                peak_new = np.array(peak_new)
                peak_new_a = np.delete(peak_new, 0)
                peak_new_b = np.delete(peak_new, -1)
                peak_new_detect = peak_new[self.detectoutliers(
                                           np.abs(peak_new_a-peak_new_b), 
                                           rate=0.5, option=1)]

                if len(peak_new_detect) == (self.busbar + 1) * self.row + 1:
                    # if len(peak_new_detect) == (4 + 1) * 6 + 1:
                    edge_y.append(peak_new_detect)
                    inx_mean = ((2 * inx + 1) * 
                                (image_thre.shape[1] / 
                                split) - 1) / 2
                    inx_x.append(inx_mean)

        edge_y = np.array(edge_y)  
            
        hlines = list(zip(*edge_y))
        hlines = np.array(hlines)
        inx_x = np.array(inx_x)
        # for lines in hlines:
        #    lines_new = self.detectoutliers(lines, option=0)
        #    while np.std(lines_new) > 10:
        #        lines_new = self.detectoutliers(lines, rate=1, option=0)

        # hb_abs = [] # all lines including busbar
        hb_abs = self.linear_regression(inx_x, hlines)
        hline_abs = []  # all lines excluding busbar
       
        # for horizonline in hlines:
        #    ab, _ = curve_fit(self.linear, inx_x, horizonline) # y = ax + b
        #    hb_abs.append(ab)

        hline_abs = [(hb_abs[(self.busbar + 1) * i], 
                      hb_abs[(self.busbar + 1) * (i + 1)]) 
                     for i in range(self.row)]
        # hline_abs = [(hb_abs[(4+1)*i],hb_abs[(4+1)*(i+1)]) for i in range(6)]
        # hline_abs = [(hb_abs[(self.busbar+2)*i],hb_abs[(self.busbar+2)*(i+1)-1]) for i in range(self.row)]

        return hline_abs

    def displace(self, line, displacement):
        c = displacement * np.sqrt((1+line[0]**2))

        return (line[0], line[1] + c)

    def correction(self, lines, displacement):  # The first line in the tuple moves to negative direction, the second to positive direction
        lines_new = []
        for couple in lines:
            minus = self.displace(couple[0], -displacement)
            plus = self.displace(couple[1], displacement)
            lines_new.append((minus, plus))

        return lines_new

    def cross(self, vline, hline):
        x = (vline[0]*hline[1]+vline[1]) / (1-vline[0]*hline[0])
        y = (hline[0]*vline[1]+hline[1]) / (1-vline[0]*hline[0])

        return (x, y)

    def perspective_transform(self, image, src, size):
        src = np.float32(src)
        dst = np.float32([(0, 0), (size, 0), (0, size), (size, size)])
        M = cv.getPerspectiveTransform(src, dst)
        
        warped = cv.warpPerspective(image, M, (size, size))
        
        return warped

    def segment_cell(self, hline_abs, vline_abs, saveplace, displacement=1, 
                     cellsize=400, option=0, image=None):
        newhline_abs = self.correction(hline_abs, displacement)
        newvline_abs = self.correction(vline_abs, displacement)

        counter = 0
        for hline_ab in newhline_abs:
            for vline_ab in newvline_abs:
                xy = []
                for hab in hline_ab:
                    for vab in vline_ab:
                        xy.append(self.cross(vab, hab))
                
                if option == 0:    
                    warped = self.perspective_transform(self.resize, xy, 400)
                elif option == 1:
                    warped = self.perspective_transform(image, xy, 400)

                counter += 1

                cv.imwrite(saveplace+'/'+str(counter)+'.jpg', warped)        
                # plt.imshow(warped, cmap='gray')
                # plt.axis('off')
                # plt.tight_layout
                # plt.savefig(saveplace+'/'+str(counter)+'.png', dpi=600, bbox_inches='tight', pad_inches=0)
                # plt.cla()
                # plt.close()        
    
    def plot_edges(self, hline_abs, vline_abs, saveplace, displacement=1):
        newhline_abs = self.correction(hline_abs, displacement)
        newvline_abs = self.correction(vline_abs, displacement)

        plt.imshow(self.resize, cmap='gray')
        x = np.array(range(self.resize.shape[1]))
        y = np.array(range(self.resize.shape[0]))

        for h_ab in newhline_abs:
            plt.plot(x, h_ab[0][0]*x+h_ab[0][1], linewidth=0.3, 
                     linestyle='--', color='r')
            plt.plot(x, h_ab[1][0]*x+h_ab[1][1], linewidth=0.3, 
                     linestyle='--', color='r')
            
        for v_ab in newvline_abs:
            plt.plot(v_ab[0][0]*y+v_ab[0][1], y, linewidth=0.3, 
                     linestyle='--', color='r')
            plt.plot(v_ab[1][0]*y+v_ab[1][1], y, linewidth=0.3, 
                     linestyle='--', color='r')
            
        # plt.savefig(saveplace, dpi=600)
