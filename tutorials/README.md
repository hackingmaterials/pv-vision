# Notebook tutorials

Here are some tutorials of using the packages, including 

1. Do perspective transform on a tilted solar module 

2. Crop out single solar cells from the solar module

3. Classify the solar cells using CNN classifier or Random Forest

4. Analyze the output results from YOLO model 

5. IoU test of semantic segmentation

6. Extract the features of the cracks from the solar cells

We suggest learning `PV-Vision` following the order below;
1. Learn preprocessing: `transform_*.ipynb`
2. Learn model training and testing: `modelhandler_*.ipynb`
3. Learn crack feature extraction: `extract_crack_features.ipynb`

:warning: Note that we are updating our tools actively, so some tutorials may be out of date. We label them as "old_xxx.ipynb" which means they are not tested using the latest `pv-vision` tool. 

:warning: The UNet model (unet_model.py) included in this folder is an example provided for testing purposes. It is originally from the [Supervisely repository](https://github.com/supervisely/supervisely/blob/master/plugins/nn/unet_v2/src/unet.py) and is licensed under the Apache License 2.0. The copyright belongs to the original author or organization. Please note that the inclusion of this file in the tutorial folder does not affect the licensing of the rest of the project, which remains under the BSD-3-Clause License.