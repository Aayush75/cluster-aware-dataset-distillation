#!/bin/bash

# ImageNet-1K with pseudo labels from TEMI clustering (1000 clusters)
# This script skips synthesis and trains directly on full dataset with pseudo labels

RUN_PRETRAIN=false  # Set to false to skip pretraining
DATASET="imagenet"
MODEL="resnet18"
EVAL_MODEL="resnet18"
DATA_ROOT="/home/ssl.distillation/WMDD/datasets/"
CSV_ROOT="/home/ssl.distillation/clustering/results/temi_imagenet-1k_1000clusters_20251224_194551/pseudo_labels/"
EXPERIMENT=1
MODEL_EXP=1
ITERATION=2000
KD_EPOCHS=200
DEBUG=false
IPC=10
GPU=0
R_WB=1
WB=false
CDA=False
PER_CLASS_BN=false

# Parse command-line arguments. All flags are optional.
# Usage: bash run_imagenet1k_pseudo.sh -x 2 -y 1 -u 0 -z 200
# -x is the experiment id. Arguments and results will be saved in ./log/{experiment}.json
# -y is the id of the teacher model (not used in pseudo-label only training)
# -u is the GPU index
# -z is the number of training epochs

while getopts ":pd:m:e:x:y:r:i:z:gc:u:wnb:a:t:A" opt; do
  case $opt in
    p) RUN_PRETRAIN=true;;
    d) DATASET="$OPTARG";;
    m) MODEL="$OPTARG";;
    e) EVAL_MODEL="$OPTARG";;
    x) EXPERIMENT="$OPTARG";;
    y) MODEL_EXP="$OPTARG";;
    r) DATA_ROOT="$OPTARG";;
    i) ITERATION="$OPTARG";;
    z) KD_EPOCHS="$OPTARG";;
    g) DEBUG=true;;
    c) IPC="$OPTARG";;
    u) GPU="$OPTARG";;
    w) WB=true;;
    n) PER_CLASS_BN=true;;
    b) R_BN="$OPTARG";;
    a) LR="$OPTARG";;
    t) R_WB="$OPTARG";;
    A) CDA=true;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1;;
  esac
done

# Test if the code can run
if [ "$DEBUG" = true ]; then
    ITERATION=2
    KD_EPOCHS=2
    IPC=2
fi

# ImageNet-1K specific settings
R_BN=0.01
LR=0.25
KD_BATCH_SIZE=100

DATA_PATH="${DATA_ROOT}imagenet_data"

start=$(date +%s%N) # %s%N for seconds and nanoseconds

# Skip pretrain, recover, and relabel for pseudo-label training
# Jump straight to training with pseudo labels

echo "=========================================="
echo "Training ImageNet-1K with Pseudo Labels"
echo "Dataset: ${DATASET}"
echo "Model: ${EVAL_MODEL}"
echo "Experiment ID: ${EXPERIMENT}"
echo "Epochs: ${KD_EPOCHS}"
echo "GPU: ${GPU}"
echo "Data Path: ${DATA_PATH}"
echo "CSV Path: ${CSV_ROOT}"
echo "=========================================="

cd ./train/
CUDA_VISIBLE_DEVICES=${GPU} python train_FKD.py \
    --dataset ${DATASET} \
    --model ${EVAL_MODEL} \
    --batch-size 256 \
    --input-size 224 \
    --epochs ${KD_EPOCHS} \
    --exp-name ${EXPERIMENT}_pseudo \
    --sgd \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --cos \
    --temperature 20 \
    --workers 8 \
    --gradient-accumulation-steps 1 \
    --train-dir ${DATA_PATH}/train \
    --val-dir ${DATA_PATH}/val \
    --fkd-path "dummy" \
    --pseudo-label-csv "${CSV_ROOT}train_image_pseudo_labels.csv" \
    --use-parquet-dataset \
    --parquet-data-dir "${DATA_PATH}/data" \
    --output-dir "./save/imagenet1k_rn18_pseudo/${EXPERIMENT}/"
cd ..

end=$(date +%s%N)
duration=$((end - start))
echo "Duration: $((duration / 1000000000)) seconds and $((duration % 1000000000)) nanoseconds."
echo "Training completed! Results saved to ./save/imagenet1k_rn18_pseudo/${EXPERIMENT}/"
