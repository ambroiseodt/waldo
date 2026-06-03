# #!/usr/bin/bash

# # This file is useful to evaluate VFMs on image classification benchmarks. It will create 
# # a dedicated tmux session on the specified device. To do so, run the following
# # command in the terminal from the root directory of the project.
# # ```bash
# # $ bash <path_to_file_folder>/eval.sh
# # ```

# DATASETS
datasets=(
    "aircraft"
    "cifar10"
    "cifar10_c-corruption-motion_blur-severity-5"
    "cifar100"
    "domainnet-sketch"
    "dtd"
    "eurosat"
    "flowers102"
    "food101"
    "pet"
)

# # ViT
# model_name="vit"
# device="cuda:0"
# patch_size=16

# # Runs
# for model_size in \
#     "base" \
#     "large" 
# do
#     session="eval_${model_name}_${model_size}"
#     tmux new-session -d -s ${session}
#     for dataset_name in ${datasets[@]}
#     do
#         run="model_size=${model_size} patch_size=${patch_size} dataset_name=${dataset_name} device=${device}"
#         command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
#         echo "Running command: ${command}"
#         tmux send-keys -t ${session} "${command}" C-m
#     done
# done

# # ViT
# model_name="vit"
# device="cuda:0"
# patch_size=14
# model_size="huge"
# batch_size=128

# # Runs
# session="eval_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# for dataset_name in ${datasets[@]}
# do
#     run="model_size=${model_size} patch_size=${patch_size} dataset_name=${dataset_name} batch_size=${batch_size} device=${device}"
#     command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
#     echo "Running command: ${command}"
#     tmux send-keys -t ${session} "${command}" C-m
# done

# # DINOv2
# model_name="dinov2"
# device="cuda:0"
# batch_size=128

# # Runs
# for model_size in \
#     "large" \
#     # "small" \
#     # "base" \
#     # "huge" 
# do
#     session="eval_${model_name}_${model_size}"
#     tmux new-session -d -s ${session}
#     for dataset_name in ${datasets[@]}
#     do
#         run="model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} device=${device}"
#         command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
#         echo "Running command: ${command}"
#         tmux send-keys -t ${session} "${command}" C-m
#     done
# done

# # DINOv3
# model_name="dinov3"
# device="cuda:0"
# batch_size=128

# # Runs
# for model_size in \
#     "vitl16" \
#     # "vits16" \
#     # "vitb16" \
#     # "vit7b16"
# do
#     session="eval_${model_name}_${model_size}"
#     tmux new-session -d -s ${session}
#     for dataset_name in ${datasets[@]}
#     do
#         run="model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} device=${device}"
#         command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
#         echo "Running command: ${command}"
#         tmux send-keys -t ${session} "${command}" C-m
#     done
# done

# # AIMv2
# model_name="aimv2"
# device="cuda:0"
# batch_size=64

# # Runs
# for model_size in \
#     "1B" \
#     # "large" \
#     # "huge" \
#     # "3B"
# do
#     session="eval_${model_name}_${model_size}"
#     tmux new-session -d -s ${session}
#     for dataset_name in ${datasets[@]}
#     do
#         run="model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} device=${device}"
#         command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
#         echo "Running command: ${command}"
#         tmux send-keys -t ${session} "${command}" C-m
#     done
# done
