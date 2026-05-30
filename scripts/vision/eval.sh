#!/usr/bin/bash

# This file is useful to evaluate VFMs on image classification benchmarks. It will create 
# a dedicated tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/eval.sh
# ```

# # ViT
# model_name="vit"
# device="cuda:2"

# # Runs
# for model_size in \
#     "base" \
#     # "large" \
#     # "huge" 
# do
#     session="eval_${model_name}_${model_size}"
#     tmux new-session -d -s ${session}
#     run="model_size=${model_size} device=${device}"
#     command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
#     echo "Running command: ${command}"
#     tmux send-keys -t ${session} "${command}" C-m
# done

# DINOv3
model_name="dinov3"
device="cuda:2"

# Runs
for model_size in \
    "vits16" \
    # "vitb16" \
    # "vit7b16" 
do
    session="eval_${model_name}_${model_size}"
    tmux new-session -d -s ${session}
    run="model_size=${model_size} device=${device}"
    command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
    echo "Running command: ${command}"
    tmux send-keys -t ${session} "${command}" C-m
done