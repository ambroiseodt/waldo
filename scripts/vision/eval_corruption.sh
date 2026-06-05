#!/usr/bin/bash

# This file is useful to evaluate VFMs on domain generalization benchmarks. It will create 
# a dedicated tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/eval_corruption.sh
# ```

# CORRUPTIONS
corruptions="[contrast,gaussian_noise,motion_blur,snow,speckle_noise]" 

# SEVERITIES
severities="[1,3,5]" 

# # ViT
# model_name="vit"
# device="cuda:4"
# patch_size=16

# # Runs
# for model_size in \
#     large \
#     # "base" 
# do
#     session="eval_corruption_${model_name}_${model_size}"
#     tmux new-session -d -s ${session}
#     run="model_name=${model_name} model_size=${model_size} patch_size=${patch_size} corruptions=${corruptions} severities=${severities} device=${device}"
#     command="python -m apps.vision.eval_corruption config=apps/vision/configs/corruption.yaml ${run}"
#     echo "Running command: ${command}"
#     tmux send-keys -t ${session} "${command}" C-m
# done

# # ViT
# model_name="vit"
# device="cuda:4"
# patch_size=14
# model_size="huge"
# batch_size=128

# # Runs
# session="eval_corruption_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# run="model_name=${model_name} model_size=${model_size} patch_size=${patch_size} corruptions=${corruptions} severities=${severities} batch_size=${batch_size} device=${device}"
# command="python -m apps.vision.eval_corruption config=apps/vision/configs/corruption.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # DINOv2
# model_name="dinov2"
# device="cuda:4"
# batch_size=128

# # Runs
# for model_size in \
#     base \
#     # "small" \
#     # "large" \
#     # "giant" 
# do
#     session="eval_corruption_${model_name}_${model_size}"
#     tmux new-session -d -s ${session}
#     run="model_name=${model_name} model_size=${model_size}  corruptions=${corruptions} severities=${severities} batch_size=${batch_size} device=${device}"
#     command="python -m apps.vision.eval_corruption config=apps/vision/configs/corruption.yaml ${run}"
#     echo "Running command: ${command}"
#     tmux send-keys -t ${session} "${command}" C-m
# done

# # DINOv3
# model_name="dinov3"
# device="cuda:4"
# batch_size=128

# # Runs
# for model_size in \
#     "vitb16" \
#     # "vits16" \
#     # "vitl16" \
#     # "vit7b16" 
# do
#     session="eval_corruption_${model_name}_${model_size}"
#     tmux new-session -d -s ${session}
#     run="model_name=${model_name} model_size=${model_size}  corruptions=${corruptions} severities=${severities} batch_size=${batch_size} device=${device}"
#     command="python -m apps.vision.eval_corruption config=apps/vision/configs/corruption.yaml ${run}"
#     echo "Running command: ${command}"
#     tmux send-keys -t ${session} "${command}" C-m
# done

# # AIMv2
# model_name="aimv2"
# device="cuda:4"
# batch_size=128

# # Runs
# for model_size in \
#     "huge" \
#     # "large" \
#     # "1B" \
#     # "3B" 
# do
#     session="eval_corruption_${model_name}_${model_size}"
#     tmux new-session -d -s ${session}
#     run="model_name=${model_name} model_size=${model_size}  corruptions=${corruptions} severities=${severities} batch_size=${batch_size} device=${device}"
#     command="python -m apps.vision.eval_corruption config=apps/vision/configs/corruption.yaml ${run}"
#     echo "Running command: ${command}"
#     tmux send-keys -t ${session} "${command}" C-m
# done