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
    "cifar10_c-corruption-contrast-severity-5"
    "cifar10_c-corruption-motion_blur-severity-5"
    "cifar10_c-corruption-speckle_noise-severity-5"
    "cifar100"
    "domainnet-sketch"
    "dtd"
    "eurosat"
    "flowers102"
    "food101"
    "pet"
    "domainnet-clipart"
    "domainnet-painting"
    "domainnet-quickdraw"
    "domainnet-infograph"
    "rxrx1"
    "iwildcam"
)

# # ViT
# model_name="vit"
# device="cuda:7"
# patch_size=16

# # Runs
# for model_size in \
#     "large" \
#     # "base" 
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
# device="cuda:7"
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
# device="cuda:7"
# batch_size=32

# # Runs
# for model_size in \
#     "large" \
#     # "small" \
#     # "base" \
#     # "giant" 
# do
#     session="eval_${model_name}_${model_size}"
#     tmux new-session -d -s ${session}
#     for dataset_name in ${datasets[@]}
#     do
#         run="model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} device=${device}"
#         command="OMP_NUM_THREADS=56 MKL_NUM_THREADS=56 taskset -c 56-111 python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
#         echo "Running command: ${command}"
#         tmux send-keys -t ${session} "${command}" C-m
#     done
# done

# # DINOv3
# model_name="dinov3"
# device="cuda:7"
# batch_size=8

# # Runs
# for model_size in \
#     "vit7b16" \
#     "vitl16" \
#     "vits16" \
#     "vitb16" 
# do
#     session="eval_painting_${model_name}_${model_size}"
#     tmux new-session -d -s ${session}
#     for dataset_name in ${datasets[@]}
#     do
#         run="model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} device=${device}"
#         command="OMP_NUM_THREADS=56 MKL_NUM_THREADS=56 taskset -c 56-111 python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
#         echo "Running command: ${command}"
#         tmux send-keys -t ${session} "${command}" C-m
#     done
# done

# # AIMv2
# model_name="aimv2"
# device="cuda:7"
# batch_size=8

# # Runs
# for model_size in \
#     "3B" \
#     "huge" \
#     "1B" \
#     "large" 
# do
#     session="eval_quickdraw_${model_name}_${model_size}"
#     tmux new-session -d -s ${session}
#     for dataset_name in ${datasets[@]}
#     do
#         run="model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} device=${device}"
#         command="OMP_NUM_THREADS=56 MKL_NUM_THREADS=56 taskset -c 56-111 python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
#         echo "Running command: ${command}"
#         tmux send-keys -t ${session} "${command}" C-m
#     done
# done
