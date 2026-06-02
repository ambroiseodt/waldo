#!/usr/bin/bash

# This file is useful to evaluate VFMs on image classification benchmarks. It will create 
# a dedicated tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/eval.sh
# ```

# DATASETS
datasets=(
    "aircraft"
    "cifar10"
    "cifar100"
    "domainnet-sketch"
    "dtd"
    "eurosat"
    "flowers102"
    "food101"
    "imagenetc-corruption-motion_blur-severity-3"
    "pet"
)

# ViT
model_name="vit"
device="cuda:4"

# Runs
for model_size in \
    "base" \
    # "large" \
    # "huge" 
do
    session="eval_${model_name}_${model_size}"
    tmux new-session -d -s ${session}
    for dataset_name in ${datasets[@]}
    do
        run="model_size=${model_size} dataset_name=${dataset_name} device=${device}"
        command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
        echo "Running command: ${command}"
        tmux send-keys -t ${session} "${command}" C-m
    done
done
