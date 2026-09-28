#!/usr/bin/bash

# This file is useful to evaluate VFMs on Cifar10-C. It will create 
# a dedicated tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/eval_cifar10_c.sh
# ```

# CORRUPTIONS
corruptions="[contrast,gaussian_noise,motion_blur,snow,speckle_noise]" 

# SEVERITIES
severities="[1,3,5]" 

# ViT
model_name="vit"
device="cuda:4"
patch_size=16

# Runs
for model_size in \
    "base" \
    large 
do
    session="eval_cifar10_c_${model_name}_${model_size}"
    tmux new-session -d -s ${session}
    run="model_name=${model_name} model_size=${model_size} patch_size=${patch_size} corruptions=${corruptions} severities=${severities} device=${device}"
    command="python -m apps.vision.eval_cifar10_c config=apps/vision/configs/cifar10_c.yaml ${run}"
    echo "Running command: ${command}"
    tmux send-keys -t ${session} "${command}" C-m
done

# ViT
model_name="vit"
device="cuda:4"
patch_size=14
model_size="huge"
batch_size=128

# Runs
session="eval_cifar10_c_${model_name}_${model_size}"
tmux new-session -d -s ${session}
run="model_name=${model_name} model_size=${model_size} patch_size=${patch_size} corruptions=${corruptions} severities=${severities} batch_size=${batch_size} device=${device}"
command="python -m apps.vision.eval_cifar10_c config=apps/vision/configs/cifar10_c.yaml ${run}"
echo "Running command: ${command}"
tmux send-keys -t ${session} "${command}" C-m

# DINOv2
model_name="dinov2"
device="cuda:3"
batch_size=8

# Runs
for model_size in \
    "small" \
    "base" \
    "large" \
    "giant" 
do
    session="eval_cifar10_c_${model_name}_${model_size}"
    tmux new-session -d -s ${session}
    run="model_name=${model_name} model_size=${model_size}  corruptions=${corruptions} severities=${severities} batch_size=${batch_size} device=${device}"
    command="python -m apps.vision.eval_cifar10_c config=apps/vision/configs/cifar10_c.yaml ${run}"
    echo "Running command: ${command}"
    tmux send-keys -t ${session} "${command}" C-m
done

# DINOv3
model_name="dinov3"
device="cuda:0"
batch_size=8

# Runs
for model_size in \
    "vits16" \
    "vitb16" \
    "vitl16" \
    "vit7b16" 
do
    session="eval_cifar10_c_${model_name}_${model_size}"
    tmux new-session -d -s ${session}
    run="model_name=${model_name} model_size=${model_size}  corruptions=${corruptions} severities=${severities} batch_size=${batch_size} device=${device}"
    command="python -m apps.vision.eval_cifar10_c config=apps/vision/configs/cifar10_c.yaml ${run}"
    echo "Running command: ${command}"
    tmux send-keys -t ${session} "${command}" C-m
done

# AIMv2
model_name="aimv2"
device="cuda:4"
batch_size=8

# Runs
for model_size in \
    "large" \
    "huge" \
    "1B" \
    "3B" 
do
    session="eval_cifar10_c_${model_name}_${model_size}"
    tmux new-session -d -s ${session}
    run="model_name=${model_name} model_size=${model_size}  corruptions=${corruptions} severities=${severities} batch_size=${batch_size} device=${device}"
    command="python -m apps.vision.eval_cifar10_c config=apps/vision/configs/cifar10_c.yaml ${run}"
    echo "Running command: ${command}"
    tmux send-keys -t ${session} "${command}" C-m
done