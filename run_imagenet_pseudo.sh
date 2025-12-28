#!/bin/bash

RUN_PRETRAIN=false  # Set to false to skip pretraining
DATASET="imagenet"
MODEL="resnet18"
EVAL_MODEL="resnet18"
DATA_ROOT="/home/ssl.disitillation/WMDD/datasets/"
EXPERIMENT=1
MODEL_EXP=1
ITERATION=2000
KD_EPOCHS=300
DEBUG=false
IPC=10
GPU=0
R_WB=1
WB=false
CDA=False
PER_CLASS_BN=false

# Parse command-line arguments. All flags are optional.
# Usage: bash run_imagenet_pseudo.sh -x 2 -y 1 -u 0 -c 10 -r /home/user/data/ -n -w -b 3.0
# -x is the experiment id. Arguments and results will be saved in ./log/{experiment}.json
# If -p is included, it pretrains a model from scratch and saves it with the id given by '-y'.
# -y is the id of the teacher model under the (dataset, model) category. Make sure the model exists if '-p' is not set

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
: ${R_BN:=0.01}
: ${LR:=0.25}

# Handle batch size for ImageNet-1K
KD_BATCH_SIZE=128

DATA_PATH="${DATA_ROOT}imagenet_data"

start=$(date +%s%N) # %s%N for seconds and nanoseconds

# Skip pretrain, recover, and relabel for pseudo-label training
# Jump straight to training with pseudo labels

# Uncomment below if you want to pretrain a teacher model
# if [ "${RUN_PRETRAIN}" = true ]; then
#     cd ./pretrain/
#     CUDA_VISIBLE_DEVICES=${GPU} python pretrain.py \
#         --dataset ${DATASET} \
#         --model ${MODEL} \
#         --data-path ${DATA_PATH} \
#         --exp-name ${MODEL_EXP} \
#         --opt sgd \
#         --lr 0.025 \
#         --wd 1e-4 \
#         --batch-size 256 \
#         --lr-scheduler cosine \
#         --epochs 90 \
#         --augmix-severity 0 \
#         --ra-magnitude 0
#     cd ..
# fi

# Uncomment below if you want to run data synthesis (distillation)
# cd ./recover/
# if [ "${PER_CLASS_BN}" = true ]; then
#     SYNTHESIS_SCRIPT="data_synthesis_new.py"
# else
#     SYNTHESIS_SCRIPT="data_synthesis.py"
# fi

# CUDA_VISIBLE_DEVICES=${GPU} python $SYNTHESIS_SCRIPT \
#     --dataset ${DATASET} \
#     --model ${MODEL} \
#     --ckpt-path "../pretrain/output/${DATASET}_${MODEL}/${MODEL_EXP}/model_89.pth" \
#     --real-data-path ${DATA_PATH} \
#     --exp-name ${EXPERIMENT} \
#     --wb ${WB} \
#     --ipc ${IPC} \
#     --batch-size 100 \
#     --lr ${LR} \
#     --iteration ${ITERATION} \
#     --l2-scale 0 \
#     --tv-l2 0 \
#     --r-bn ${R_BN} \
#     --verifier \
#     --store-best-images \
#     --cda ${CDA} \
#     --per-class-bn ${PER_CLASS_BN} \
#     --weight-wb false \
#     --r-wb ${R_WB}
# cd ..

# Uncomment below if you want to run soft label generation
# cd ./relabel/
# CUDA_VISIBLE_DEVICES=${GPU} python generate_soft_label.py \
#     --dataset ${DATASET} \
#     --model ${MODEL} \
#     --exp-name ${EXPERIMENT} \
#     --ckpt-path "../pretrain/output/${DATASET}_${MODEL}/${MODEL_EXP}/model_89.pth" \
#     --batch-size ${KD_BATCH_SIZE} \
#     --epochs ${KD_EPOCHS} \
#     --workers 8 \
#     --fkd-seed 42 \
#     --input-size 224 \
#     --min-scale-crops 0.08 \
#     --max-scale-crops 1 \
#     --use-fp16 \
#     --fkd-path FKD_cutmix_fp16 \
#     --mode 'fkd_save' \
#     --mix-type 'cutmix' \
#     --data "../recover/syn_data/${DATASET}/${EXPERIMENT}"
# cd ..

end=$(date +%s%N)
duration=$((end - start))
echo "Duration: $((duration / 1000000000)) seconds and $((duration % 1000000000)) nanoseconds."

cd ./train/
CUDA_VISIBLE_DEVICES=${GPU} python train_FKD.py \
    --dataset ${DATASET} \
    --model ${EVAL_MODEL} \
    --batch-size ${KD_BATCH_SIZE} \
    --input-size 224 \
    --epochs ${KD_EPOCHS} \
    --exp-name ${EXPERIMENT}_pseudo \
    --sgd \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --cos \
    --temperature 20 \
    --workers 8 \
    --gradient-accumulation-steps 1 \
    --train-dir ${DATA_PATH}/train \
    --val-dir ${DATA_PATH}/val \
    --fkd-path "dummy" \
    --pseudo-label-csv "../train_image_pseudo_labels.csv" \
    --output-dir "./save/imagenet_rn18_pseudo/${EXPERIMENT}/"
cd ..
