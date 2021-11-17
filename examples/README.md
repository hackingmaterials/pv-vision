# Examples for testing 
This folder contains images and annotation files that can be used to test `pv-vision` modules. 
:warning: The data here can only be used to practice `pv-vision`. Any other usage is **NOT** allowed.

## transform_seg
This folder contains original EL images in "rgb" or "grayscale". Annotations are also included for perspective transformation in `unet_ann` folder. The colormap of transforming RGB images into Grayscale images is not provided due to NDA.

## object_detection
This folder contains the solar module images that have been perspective transformed. Also manual annotations that indicate the positions of defective cells are provided in `yolo_manual_ann` folder

## cell_classification
This folder contains images of single solar cells which are cropped out from solar modules. Those cropped cells are classified based on the manual labels from `../object_detection/yolo_manual_ann`
