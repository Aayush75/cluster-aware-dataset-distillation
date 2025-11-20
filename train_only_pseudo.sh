#!/bin/bash

# Train ONLY - skip synthesis and relabel, use existing experiment 1 data with pseudo labels

EXPERIMENT=1
GPU=0
EPOCHS=300
BATCH_SIZE=100

cd ./train/
CUDA_VISIBLE_DEVICES=${GPU} python train_FKD.py \
    --dataset imagenette \
    --model resnet18 \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --exp-name ${EXPERIMENT}_pseudo \
    --cos \
    --temperature 20 \
    --workers 8 \
    --gradient-accumulation-steps 1 \
    --train-dir "../recover/syn_data/imagenette/${EXPERIMENT}" \
    --val-dir ~/WMDD/datasets/imagenette/val \
    --fkd-path "../relabel/FKD_cutmix_fp16/imagenette/${EXPERIMENT}" \
    --mix-type 'cutmix' \
    --pseudo-label-csv "../train_image_pseudo_labels_imagenette.csv" \
    --output-dir "./save/final_rn18_fkd/${EXPERIMENT}_pseudo/"
cd ..
