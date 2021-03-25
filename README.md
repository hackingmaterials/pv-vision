# pv_vision
 Image analysis of cracks on solar cell

This package allows you to analyze the electroluminescene images of PV module. The functions of this package include image segmentation, crack detection, defective cells classification and crack-power loss correlation. 

Corresponding neural network model can be downloaded here: 
https://drive.google.com/drive/folders/1Xxy2QfqJSXIVszi2vwIFnwPb7xDjZyfG?usp=sharing

Currently the model weights are: 
1. "detect_cracks_unet_v3" is used for predicting the pixels that belong to cracks, busbars, etc. using semantic segmentation.
2. "detect_defective_cell_yolo_v6" is used for detecing defective cells in the solar modules using objective detect. Corresponding classfication codes can be found in "defective_cell_detection".
3. "detect_contour_unet_v2" is used for perspective transformation of solar module images using semantic segmantation.
4. "defective_cell_classification" is used to classify the solar cells based on the type of its defects. There are four kinds of models in this folder, which are random forest(RF) model, lenet model, resnet18 model and resnet50 model.