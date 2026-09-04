# #!/usr/bin/bash

# # This file is useful to evaluate VFMs on image classification benchmarks. It will create 
# # a dedicated tmux session on the specified device. To do so, run the following
# # command in the terminal from the root directory of the project.
# # ```bash
# # $ bash <path_to_file_folder>/eval.sh
# # ```

# # DATASETS
# datasets=(
#     "pcam"
# )

# # DINOv3
# model_name="dinov3"
# device="cuda:7"
# batch_size=8

# # Runs
# for model_size in \
#     "vit7b16" \
#     "vits16" \
#     "vitb16" \
#     "vitl16" 
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


# # DATASETS
# datasets=(
#     "camelyon17"
#     "iwildcam"
#     "fmow"
#     "pcam"
#     "rxrx1"
# )

# # DINOv3
# model_name="dinov3"
# device="cuda:7"
# batch_size=8
# model_size="vitl16"

# # Runs
# for dataset_name in ${datasets[@]}
# do
#     session="eval_${dataset_name}_${model_name}_${model_size}"
#     tmux new-session -d -s ${session}
#     run="model_size=${model_size} dataset_name=${dataset_name} batch_size=${batch_size} device=${device}"
#     command="python -m apps.vision.eval config=apps/vision/configs/${model_name}.yaml ${run}"
#     echo "Running command: ${command}"
#     tmux send-keys -t ${session} "${command}" C-m
# done