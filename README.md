 # PV-Vision

[![GitHub license](https://img.shields.io/github/license/hackingmaterials/pv-vision)](https://github.com/hackingmaterials/pv-vision/blob/main/LICENSE)
[![Requires Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg?label=Requires%20Python)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/pv-vision)](https://pypi.org/project/pv-vision/)
[![DOI](https://zenodo.org/badge/253107508.svg)](https://zenodo.org/badge/latestdoi/253107508)


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

3. To enable CUDA and GPU acceleration, install [Pytorch with cudatoolkit](https://pytorch.org/get-started/locally/)

## Citation
**Please cite our papers:**

Automatic defect identification pipeline:
```
@article{chen2022automated,
  title={Automated defect identification in electroluminescence images of solar modules},
  author={Chen, Xin and Karin, Todd and Jain, Anubhav},
  journal={Solar Energy},
  volume={242},
  pages={20--29},
  year={2022},
  publisher={Elsevier}
}
``` 

Automatic crack segmentation and feature extraction:
```
# Crack segmentation paper
@article{chen2023automatic,
  title={Automatic Crack Segmentation and Feature Extraction in Electroluminescence Images of Solar Modules},
  author={Chen, Xin and Karin, Todd and Libby, Cara and Deceglie, Michael and Hacke, Peter and Silverman, Timothy J and Jain, Anubhav},
  journal={IEEE Journal of Photovoltaics},
  year={2023},
  publisher={IEEE}
}
```

**We also published our data set:**
```
# Crack segmentation dataset
@misc{chen2022benchmark,
  title={A Benchmark for Crack Segmentation in Electroluminescence Images},
  doi={10.21948/1871275},
  url={https://datahub.duramat.org/dataset/crack-segmentation},
  author={Chen, Xin and Karin, Todd and Libby, Cara and Deceglie, Michael and Hacke, Peter and Silverman, Timothy and Gabor, Andrew and Jain, Anubhav},
  year={2022},
}
```

**If you want to cite the PV-Vision:**
```
@misc{PV-Vision,
  doi={10.5281/ZENODO.6564508},
  url={https://github.com/hackingmaterials/pv-vision},
  author={Chen, Xin},
  title={pv-vision},
  year={2022},
  copyright={Open Access}
}
```

## Overview

This package allows you to analyze electroluminescene (EL) images of photovoltaics (PV) modules. The methods provided in this package include module transformation, cell segmentation, crack segmentation, defective cells identification, etc. Future work will include photoluminescence image analysis, image denoising, barrel distortion fixing, etc. 

You can either use the package `pv_vision` and write your own codes following the instruction in tutorials, or you can directly run our `pipeline.sh` to do automated defects indentification. When `pipeline.sh` is used, `YOLO` model will be applied to do predictions in default. The output will give you the analysis from the model.

Our trained neural network models can be downloaded [here](https://datahub.duramat.org/dataset/neural-network-weights).

Currently the model weights are: 

1. Folder "crack_segmentation" is used for predicting the pixels that belong to cracks, busbars, etc. using semantic segmentation.

2. Folder "defect_detection" is used to do object detection of defective cells. 

3. Folder "cell_classification" is used to do cell classification. 

4. Folder "module_segmentation" is used for perspective transformation of solar module images using semantic segmantation. It will predict the contour of field module images

## Analyze data
The tutorials of using `PV-Vision` can be found in folder `tutorials`. The tutorials cover *perspective transformation*, *cell segmentation*, *model inference* and *model output analysis*. 

## Public dataset
We published one of our datasets as a benchmark for crack segmentation. Images and annotations can be found on [DuraMat datahub](https://datahub.duramat.org/dataset/crack-segmentation)

## Deploy models
There are three ways to deply our deep learning models:

### 1. Use Python
Check tutorials of ``modelhandler.py``. This tool allows you to train your own deep learning models.
```
from pv_vision.nn import ModelHandler
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

After you have deployed the models, run our pipeline script to get the predictions. **Note that this pipeline was only designed for object detection and doesn't have active maintenance currently** Check our tutorials about how to do crack analysis.

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
3. ~~We also developed algoritms of extracting cracks from solar cells. We will integrate the algorithms with `PV-Vision`.~~ **Done**
4. ~~We want to predict the worst degradation amount based on the existing crack pattern. This will also be integrated into `PV-Vision`.~~ **Done**
5. ~~Add neural network modules~~ **Done**
6. ~~Add result analysis~~ **Done**