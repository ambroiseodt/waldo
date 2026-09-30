#!/usr/bin/bash

# This file is useful to run the ablations of VFMs on image classification benchmarks. It will
# create a dedicated tmux session per ablation and model on the specified device. The runs are
# saved in savings/<log_dir>/ to keep them separate from the main runs. To do so, run the
# following command in the terminal from the root directory of the project.
# ```bash
# $ bash scripts/vision/ablation.sh
# ```

# DATASETS
datasets=(
    "cifar10"
    "domainnet-sketch"
    "domainnet-quickdraw"
)
device="cuda:0"

# Registers: DINOv2-L with registers, all layers and modules (CLS and registers excluded from the pooling)
log_dir="register"
model_name="dinov2"
model_size="large-registers"
batch_size=32

# Runs
session="${log_dir}_${model_name}_${model_size}"
tmux new-session -d -s ${session}
for dataset_name in ${datasets[@]}
do
    run="model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} device=${device} log_dir=${log_dir}"
    command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
    echo "Running command: ${command}"
    tmux send-keys -t ${session} "${command}" C-m
done

# CLS concatenation: CLS token concatenated with the mean over patch tokens, all layers and modules
# (AIMv2 has no CLS token)
log_dir="cls_concat"
pooling="all"

# Runs
for model in \
    "vit large 512" \
    "dinov2 large 32" \
    "dinov3 vitl16 8"
do
    read model_name model_size batch_size <<< ${model}
    session="${log_dir}_${model_name}_${model_size}"
    tmux new-session -d -s ${session}
    for dataset_name in ${datasets[@]}
    do
        run="model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} pooling=${pooling} device=${device} log_dir=${log_dir}"
        command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
        echo "Running command: ${command}"
        tmux send-keys -t ${session} "${command}" C-m
    done
done

# Final norm: output of the final norm of the last layer (standard output of the model)
log_dir="final_norm"

# Runs
for model in \
    "vit large 512" \
    "dinov2 large 32" \
    "dinov3 vitl16 8" \
    "aimv2 large 8"
do
    read model_name model_size batch_size <<< ${model}
    session="${log_dir}_${model_name}_${model_size}"
    tmux new-session -d -s ${session}
    for dataset_name in ${datasets[@]}
    do
        run="model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} layers=[-1] modules=[rc2norm] device=${device} log_dir=${log_dir}"
        command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
        echo "Running command: ${command}"
        tmux send-keys -t ${session} "${command}" C-m
    done
done
