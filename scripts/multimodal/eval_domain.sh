#!/usr/bin/bash

# This file is useful to evaluate VFMs on domain generalization benchmarks. It will create 
# a dedicated tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/eval_domain.sh
# ```

# DOMAINS
domains="[clipart,infograph,painting,quickdraw,sketch]" 

# # Clip
# model_name="clip"
# device="cuda:4"
# model_size="base"
# patch_size=16
# batch_size=16

# # Runs
# session="eval_domain_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# run="model_name=${model_name} model_size=${model_size} patch_size=${patch_size} domains=${domains} batch_size=${batch_size} device=${device}"
# command="python -m apps.multimodal.eval_domain config=apps/multimodal/configs/domain.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m


# # Clip
# model_name="clip"
# device="cuda:4"
# model_size="large"
# patch_size=14
# batch_size=16

# # Runs
# session="eval_domain_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# run="model_name=${model_name} model_size=${model_size} patch_size=${patch_size} domains=${domains} batch_size=${batch_size} device=${device}"
# command="python -m apps.multimodal.eval_domain config=apps/multimodal/configs/domain.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # Siglip2
# model_name="siglip2"
# device="cuda:4"
# model_size="base"
# patch_size=16
# batch_size=16

# # Runs
# session="eval_domain_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# run="model_name=${model_name} model_size=${model_size} patch_size=${patch_size} domains=${domains} batch_size=${batch_size} device=${device}"
# command="python -m apps.multimodal.eval_domain config=apps/multimodal/configs/domain.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # Siglip2
# model_name="siglip2"
# device="cuda:4"
# model_size="large"
# patch_size=16
# resolution=256
# batch_size=16

# # Runs
# session="eval_domain_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# run="model_name=${model_name} model_size=${model_size} patch_size=${patch_size} resolution=${resolution} domains=${domains} batch_size=${batch_size} device=${device}"
# command="python -m apps.multimodal.eval_domain config=apps/multimodal/configs/domain.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # Siglip2
# model_name="siglip2"
# device="cuda:4"
# model_size="giant-opt"
# patch_size=16
# resolution=256
# batch_size=16

# # Runs
# session="eval_domain_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# run="model_name=${model_name} model_size=${model_size} patch_size=${patch_size} resolution=${resolution} domains=${domains} batch_size=${batch_size} device=${device}"
# command="python -m apps.multimodal.eval_domain config=apps/multimodal/configs/domain.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m