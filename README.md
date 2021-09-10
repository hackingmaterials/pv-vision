# pv_vision

 Image analysis of cracks on solar cell

This package allows you to analyze the electroluminescene images of PV module. The functions of this package include image segmentation, crack detection, defective cells classification and crack-power loss correlation. 

Corresponding neural network model can be downloaded here: 
<https://drive.google.com/drive/folders/1Xxy2QfqJSXIVszi2vwIFnwPb7xDjZyfG?usp=sharing>

Currently the model weights are: 

1. In folder "crack_extraction":

   Model "detect_cracks_unet_v3" is used for predicting the pixels that belong to cracks, busbars, etc. using semantic segmentation.

2. In folder "defective_cell_detection":

    Model "yolo_rgb" and "yolo_grayscale" are used for detecting defective cells in the solar modules using objective detection. The module images don't need to be cropped in advance. "yolo_rgb" can be applied to rgb images and "yolo_grayscale" can be applied on grayscale images.

    Models in "classifiers" are used to classify the solar cells based on the type of its defects. There are four kinds of models in this folder, which are random forest(RF) model, resnet18 model, resnet50 model and resnet152 model. "resnet_rgb" can be applied to rgb images and "resnet_grayscale" can be applied on grayscale images. The module images need to be cropped into single solar cells in advance.

3. In folder "perspective_transform":

    Model "detect_contour_unet_rgb" and "detect_contour_unet_grayscale" are used for perspective transformation of solar module images using semantic segmantation. "rgb" can be applied to rgb images and "grayscale" can be applied on grayscale images.