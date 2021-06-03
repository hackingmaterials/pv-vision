# pv_vision

 Image analysis of cracks on solar cell

This package allows you to analyze the electroluminescene images of PV module. The functions of this package include image segmentation, crack detection, defective cells classification and crack-power loss correlation. 

Corresponding neural network model can be downloaded here: 
<https://drive.google.com/drive/folders/1Xxy2QfqJSXIVszi2vwIFnwPb7xDjZyfG?usp=sharing>

Currently the model weights are: 

1. In folder "crack_extraction":

   Model "detect_cracks_unet_v3" is used for predicting the pixels that belong to cracks, busbars, etc. using semantic segmentation.

2. In folder "defective_cell_detection":

    Model "detect_defective_cell_yolo_v6" and "detect_defective_cell_yolo_v8_1" are used for detecting defective cells in the solar modules using objective detection. Corresponding classfication codes can be found in "defective_cell_detection". The module images don't need to be cropped in advance. "v6" passed the spotcheck of a company and is used in industry currently. "v8_1" is the version used in the paper.

    Models "detect_defective_cell_classifiers" are used to classify the solar cells based on the type of its defects. There are four kinds of models in this folder, which are random forest(RF) model, lenet model, resnet18 model and resnet50 model. The module images need to be cropped into single solar cells in advance.

3. In folder "perspective_transform":

    Model "detect_contour_unet_v3_1" and "detect_contour_unet_v4" are used for perspective transformation of solar module images using semantic segmantation. "v3_1" requires the module images to align the long side horizontally while "v4" doesn't have requirement for the orientation of the images.
