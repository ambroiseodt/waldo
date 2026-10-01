#!/usr/bin/bash

# This file is useful to evaluate LLMs on the MTEB benchmark. It will create 
# a dedicated tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash scripts/language/eval.sh
# ```

# # Gemma3
# model_name="gemma3"
# device="cuda:0"
# model_size="1b"

# # Runs
# session="eval_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# run="model_size=${model_size} device=${device}"
# command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # Gemma3
# model_name="gemma3"
# device="cuda:0"
# model_size="4b"
# batch_size=4

# # Runs
# session="eval_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# run="model_size=${model_size} batch_size=${batch_size} device=${device}"
# command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # Gemma3
# model_name="gemma3"
# device="cuda:0"
# model_size="12b"
# batch_size=4

# # Runs
# session="eval_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# run="model_size=${model_size} batch_size=${batch_size} device=${device}"
# command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # Llama3
# model_name="llama3"
# device="cuda:0"
# model_size="1B"
# batch_size=1

# # Runs
# session="eval_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# run="model_size=${model_size} batch_size=${batch_size} device=${device}"
# command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # Llama3
# model_name="llama3"
# device="cuda:0"
# model_size="3B"
# batch_size=1

# # Runs
# session="eval_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# run="model_size=${model_size} batch_size=${batch_size} device=${device}"
# command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # Llama3
# model_name="llama3"
# device="cuda:0"
# model_size="8B"

# # Runs
# session="eval_${model_name}_${model_size}"
# tmux new-session -d -s ${session}
# run="model_size=${model_size} device=${device}"
# command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # Qwen3
# model_name="qwen3"
# device="cuda:0"
# model_size="1.7B"

# # Clean model size for session creation (replace '.' with '_')
# clean_size=$(echo ${model_size} | tr '.' '_')
# session="eval_${model_name}_${clean_size}"
# tmux new-session -d -s ${session}
# run="model_size=${model_size} device=${device}"
# command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # Qwen3
# model_name="qwen3"
# device="cuda:0"
# model_size="4B"
# batch_size=1

# # Clean model size for session creation (replace '.' with '_')
# clean_size=$(echo ${model_size} | tr '.' '_')
# session="eval_${model_name}_${clean_size}"
# tmux new-session -d -s ${session}
# run="model_size=${model_size} batch_size=${batch_size} device=${device}"
# command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # Qwen3
# model_name="qwen3"
# device="cuda:0"
# model_size="8B"
# batch_size=1

# # Clean model size for session creation (replace '.' with '_')
# clean_size=$(echo ${model_size} | tr '.' '_')
# session="eval_${model_name}_${clean_size}"
# tmux new-session -d -s ${session}
# run="model_size=${model_size} batch_size=${batch_size} device=${device}"
# command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # Qwen3
# model_name="qwen3"
# device="cuda:0"
# model_size="1.7B-Base"
# batch_size=4

# # Clean model size for session creation (replace '.' with '_')
# clean_size=$(echo ${model_size} | tr '.' '_')
# session="eval_${model_name}_${clean_size}"
# tmux new-session -d -s ${session}
# run="model_size=${model_size} batch_size=${batch_size} device=${device}"
# command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # Qwen3
# model_name="qwen3"
# device="cuda:1"
# model_size="4B-Base"
# batch_size=8

# # Clean model size for session creation (replace '.' with '_')
# clean_size=$(echo ${model_size} | tr '.' '_')
# session="eval_${model_name}_${clean_size}"
# tmux new-session -d -s ${session}
# run="model_size=${model_size} batch_size=${batch_size} device=${device}"
# command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m

# # Qwen3
# model_name="qwen3"
# device="cuda:2"
# model_size="8B-Base"
# batch_size=4

# # Clean model size for session creation (replace '.' with '_')
# clean_size=$(echo ${model_size} | tr '.' '_')
# session="eval_${model_name}_${clean_size}"
# tmux new-session -d -s ${session}
# run="model_size=${model_size} batch_size=${batch_size} device=${device}"
# command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
# echo "Running command: ${command}"
# tmux send-keys -t ${session} "${command}" C-m