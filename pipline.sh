#!/bin/bash

#docker run -d --rm -it --runtime=nvidia -p 7000:5000 -v "$(pwd)/unet_model:/sly_task_data/model" --env GPU_DEVICE=0 supervisely/nn-unet-v2:6.0.26 python /workdir/src/rest_inference.py
#docker run -d --rm -it --runtime=nvidia -p 5000:5000 -v "$(pwd)/yolo_model:/sly_task_data/model" --env GPU_DEVICE=0 supervisely/nn-yolo-v3 python /workdir/src/rest_inference.py
mkdir -p output/unet_ann

for file in raw_images/*
do
  curl -X POST -F "image=@$file" 0.0.0.0:7000/model/inference > "output/unet_ann/$(basename "$file").json"
done

python scripts/transform_module_v2.py -i raw_images -a output/unet_ann --mask_name module_model -o output/transformation

mkdir output/yolo_ann

for file in output/transformation/transformed_images/*
do
  curl -X POST -F "image=@$file" 0.0.0.0:5000/model/inference > "output/yolo_ann/$(basename "$file").json"
done

python scripts/collect_cell_issues.py -a output/yolo_ann -n scripts/metadata/defect_name.json -o output/analysis

python scripts/highlight_defects.py -a output/yolo_ann -c scripts/metadata/defect_colors.json -i output/transformation/transformed_images -o output/analysis

python scripts/move2folders.py -c output/analysis/cell_issues.csv -i output/transformation/transformed_images -o output/analysis

