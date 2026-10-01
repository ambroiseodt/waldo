#!/usr/bin/bash

# This file is useful to run the ablations of VFMs on image classification benchmarks. It will
# create a dedicated tmux session per ablation and model on the specified device. The runs are
# saved in savings/ablations/<ablation>/ to keep them separate from the main runs. To do so, run the
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

# DINOv2-L with registers
ablation="registers"
log_dir="ablations/${ablation}"
model_name="dinov2"
model_size="large-registers"
batch_size=32
device="cuda:0"

# Runs
session="${ablation}_${model_name}_${model_size}"
tmux new-session -d -s ${session}
for dataset_name in ${datasets[@]}
do
    run="log_dir=${log_dir} model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} device=${device}"
    command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
    echo "Running command: ${command}"
    tmux send-keys -t ${session} "${command}" C-m
done

# CLS concatenation (AIMv2 has no CLS token)
ablation="cls_concat"
log_dir="ablations/${ablation}"
pooling="all"
batch_size=32
device="cuda:0"

# Runs
for model in \
    "vit large" \
    "dinov2 large" \
    "dinov3 vitl16"
do
    read model_name model_size <<< ${model}
    session="${ablation}_${model_name}_${model_size}"
    tmux new-session -d -s ${session}
    for dataset_name in ${datasets[@]}
    do
        run="log_dir=${log_dir} model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} pooling=${pooling} device=${device}"
        command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
        echo "Running command: ${command}"
        tmux send-keys -t ${session} "${command}" C-m
    done
done

# CLS token only (AIMv2 has no CLS token)
ablation="cls"
log_dir="ablations/${ablation}"
pooling="cls"
batch_size=32
device="cuda:0"

# Runs
for model in \
    "vit large" \
    "dinov2 large" \
    "dinov3 vitl16"
do
    read model_name model_size <<< ${model}
    session="${ablation}_${model_name}_${model_size}"
    tmux new-session -d -s ${session}
    for dataset_name in ${datasets[@]}
    do
        run="log_dir=${log_dir} model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} pooling=${pooling} device=${device}"
        command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
        echo "Running command: ${command}"
        tmux send-keys -t ${session} "${command}" C-m
    done
done

# Final norm
ablation="final_norm"
log_dir="ablations/${ablation}"
batch_size=32
device="cuda:0"

# Runs
for model in \
    "vit large" \
    "dinov2 large" \
    "dinov3 vitl16" \
    "aimv2 large"
do
    read model_name model_size <<< ${model}
    session="${ablation}_${model_name}_${model_size}"
    tmux new-session -d -s ${session}
    for dataset_name in ${datasets[@]}
    do
        run="log_dir=${log_dir} model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} layers=[-1] modules=[rc2norm] device=${device}"
        command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
        echo "Running command: ${command}"
        tmux send-keys -t ${session} "${command}" C-m
    done
done
