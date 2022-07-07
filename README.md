# PV-Vision
**:warning: This is a beta version tool. It is still under test for different operating environments. Please raise an issue if you cannot use it on your computer.**

**:warning: We are updating our tools actively so some of the tutorials may be out of date.**

## Installation
1. Install from source
```bash
git clone https://github.com/hackingmaterials/pv-vision.git
cd pv-vision
pip install .
```
2. Install from Pypi
```bash
pip install pv-vision
```
## Citation
Please cite our papers if you use our tool or dataset.
```
Will update soon
``` 

## Overview

Image analysis of defects on solar cells.

This package allows you to analyze electroluminescene (EL) images of PV module. The methods in this package include image segmentation, crack detection, defective cells classification and crack-power loss correlation. 

You can either use the package `pv_vision` and write your own codes following the instruction in tutorials, or you can directly run our `pipeline.sh`. When `pipeline.sh` is used, `YOLO` model will be applied to do predictions in default. The output will give you the analysis from the model.

Our trained neural network models can be downloaded [here](https://drive.google.com/drive/folders/1Xxy2QfqJSXIVszi2vwIFnwPb7xDjZyfG?usp=sharing).

Currently the model weights are: 

1. In folder "crack_extraction":

   Model "unet_oversample_low" is used for predicting the pixels that belong to cracks, busbars, etc. using semantic segmentation.

2. In folder "defective_cell\_detection":

    Model "yolo\_rgb" and "yolo\_grayscale" are used for detecting defective cells in the solar modules using objective detection. The module images don't need to be cropped in advance. "yolo\_rgb" can be applied to rgb images and "yolo\_grayscale" can be applied on grayscale images.

    Models in "classifiers" are used to classify the solar cells based on the type of its defects. There are four kinds of models in this folder, which are random forest(RF) model, resnet18 model, resnet50 model and resnet152 model. "resnet\_rgb" can be applied to rgb images and "resnet\_grayscale" can be applied on grayscale images. The module images need to be cropped into single solar cells in advance.

3. In folder "perspective_transform":

    Model "detect\_contour\_unet\_rgb" and "detect\_contour\_unet\_grayscale" are used for perspective transformation of solar module images using semantic segmantation. "rgb" can be applied to rgb images and "grayscale" can be applied on grayscale images.

## Analyze data
The tutorials of using `PV-Vision` can be found in folder `tutorials`. The tutorials cover *perspective transformation*, *cell segmentation*, *model inference* and *model output analysis*. 

## Public dataset
We published one of our datasets as a benchmark for crack segmentation. Images and annotations can be found on [DuraMat datahub](https://datahub.duramat.org/dataset/crack-segmentation)

## Deploy models
There are two ways to deply our deep learning models:

### 1. Use Python
```
Will update soon
```

### 2. Use Supervisely

Upload the model weights to [Supervisely](https://supervise.ly/) and make predictions on this website. The detailed tutorials can be found [here](https://docs.supervise.ly/) and [here](https://github.com/supervisely/supervisely/blob/master/help/tutorials/04_deploy_neural_net_as_api/deploy-model.md#method-1-through-ui).

### 3. Use docker
You can also run the models using `docker`. 

First make sure you prepare required files as stated in the following folder structure.

Then pull the images

```bash
docker pull supervisely/nn-yolo-v3
docker pull supervisely/nn-unet-v2:6.0.26
```

You should see the two images by running

```bash
docker image ls
```

Start the containers by running

```bash
docker run -d --rm -it --runtime=nvidia -p 7000:5000 -v "$(pwd)/unet_model:/sly_task_data/model" --env GPU_DEVICE=0 supervisely/nn-unet-v2:6.0.26 python /workdir/src/rest_inference.py

docker run -d --rm -it --runtime=nvidia -p 5000:5000 -v "$(pwd)/yolo_model:/sly_task_data/model" --env GPU_DEVICE=0 supervisely/nn-yolo-v3 python /workdir/src/rest_inference.py
```

Here we deploy the `UNet` to `port 7000` and `YOLO` to `port 5000`.
The path `$(pwd)/unet_model` or `$(pwd)/yolo_model` is where we store our model weights. You can download them [here](https://drive.google.com/drive/folders/1Xxy2QfqJSXIVszi2vwIFnwPb7xDjZyfG?usp=sharing).

Check if you successfully run the two dockers by running

```bash
docker container ls
```

After you have deployed the models, run our pipeline script to get the predictions.

```bash
bash pipeline.sh
```

You will find the predictions in a new folder `output`.

In general, your folder structure should be like the following. When start the containers, you need to prepare `unet_model` and `yolo_model`. When running `pipeline.sh`, You only need to prepare `pipeline`, `raw_images` where stores raw grayscale EL images, `scripts` where you need to configure the `metadata` in the parent folder `PV-pipeline`. The `output` folder will be created after you run the `pipeline.sh`.

```bash
PV-pipeline
├── unet_model
│   ├── config.json
│   └── model.pt
├── yolo_model
│   ├── config.json
│   ├── model.weights
│   └── model.cfg
├── pipeline.sh
├── raw_images
│   ├── img1.png
│   ├── img2.png
│   ├── img3.png
│   ├── img4.png
│   └── img5.png
├── scripts
│   ├── metadata
│   │   ├── defect_colors.json
│   │   └── defect_name.json
│   ├── collect_cell_issues.py
│   ├── highlight_defects.py
│   ├── move2folders.py
│   └── transform_module_v2.py
└── output
    ├── analysis
    │   ├── cell_issues.csv
    │   ├── classified_images
    │   │   ├── category1
    │   │   │   └── img1.png
    │   │   ├── category2
    │   │   │   ├── img4.png
    │   │   │   └── img2.png
    │   │   └── category3
    │   │       ├── img3.png
    │   │       └── img5.png
    │   └── visualized_images
    │       ├── img1.png
    │       ├── img2.png
    │       ├── img3.png
    │       ├── img4.png
    │       └── img5.png
    ├── transformation
    │   ├── failed_images
    │   └── transformed_images
    │       ├── img1.png
    │       ├── img2.png
    │       ├── img3.png
    │       ├── img4.png
    │       └── img5.png
    ├── unet_ann
    │   ├── img1.png.json
    │   ├── img2.png.json
    │   ├── img3.png.json
    │   ├── img4.png.json
    │   └── img5.png.json
    └── yolo_ann
        ├── img1.png.json
        ├── img2.png.json
        ├── img3.png.json
        ├── img4.png.json
        └── img5.png.json

```

## To do
1. ~~We will upload some EL images for users to practice after we get approval from our data provider.~~ **Done**
2. ~~We will improve the user experience of our tools. We will do more Object-oriented programming (OOP) in the future version.~~ **Done**
3. ~~We also developed algoritms of extracting cracks from solar cells.~~ We will integrate the algorithms with `PV-Vision`.
4. ~~We want to predict the worst degradation amount based on the existing crack pattern. This will also be integrated into `PV-Vision`.~~ **Done**