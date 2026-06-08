# #!/usr/bin/bash

# # This file is useful to evaluate VLMs on image classification benchmarks. It will create 
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

# # Clip
# model_name="clip"
# device="cuda:4"
# model_size="base"
# patch_size=16
# batch_size=16

# # Runs
# session="eval_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# for dataset_name in ${datasets[@]}
# do
#     run="model_size=${model_size} patch_size=${patch_size} dataset_name=${dataset_name} batch_size=${batch_size} device=${device}"
#     command="python -m apps.multimodal.eval config=apps/multimodal/configs/${model_name}.yaml ${run}"
#     echo "Running command: ${command}"
#     tmux send-keys -t ${session} "${command}" C-m
# done

# # Clip
# model_name="clip"
# device="cuda:0"
# model_size="large"
# patch_size=14
# batch_size=16

# # Runs
# session="eval_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# for dataset_name in ${datasets[@]}
# do
#     run="model_size=${model_size} patch_size=${patch_size} dataset_name=${dataset_name} batch_size=${batch_size} device=${device}"
#     command="python -m apps.multimodal.eval config=apps/multimodal/configs/${model_name}.yaml ${run}"
#     echo "Running command: ${command}"
#     tmux send-keys -t ${session} "${command}" C-m
# done

# # SigLip2
# model_name="siglip2"
# device="cuda:4"
# model_size="base"
# batch_size=16

# # Runs
# session="eval_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# for dataset_name in ${datasets[@]}
# do
#     run="model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} device=${device}"
#     command="python -m apps.multimodal.eval config=apps/multimodal/configs/${model_name}.yaml ${run}"
#     echo "Running command: ${command}"
#     tmux send-keys -t ${session} "${command}" C-m
# done

# # SigLip2
# model_name="siglip2"
# device="cuda:0"
# model_size="large"
# resolution=256
# batch_size=16

# # Runs
# session="eval_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# for dataset_name in ${datasets[@]}
# do
#     run="model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} resolution=${resolution} device=${device}"
#     command="python -m apps.multimodal.eval config=apps/multimodal/configs/${model_name}.yaml ${run}"
#     echo "Running command: ${command}"
#     tmux send-keys -t ${session} "${command}" C-m
# done

# # SigLip2
# model_name="siglip2"
# device="cuda:0"
# model_size="giant-opt"
# resolution=256
# batch_size=16

# # Runs
# session="eval_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# for dataset_name in ${datasets[@]}
# do
#     run="model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} resolution=${resolution} device=${device}"
#     command="python -m apps.multimodal.eval config=apps/multimodal/configs/${model_name}.yaml ${run}"
#     echo "Running command: ${command}"
#     tmux send-keys -t ${session} "${command}" C-m
# done