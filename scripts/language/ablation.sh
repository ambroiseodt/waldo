#!/usr/bin/bash

# This file is useful to run the ablations of LLMs on the MTEB benchmark. It will create a
# dedicated tmux session on the specified device, where the models are evaluated one after
# the other. The runs are saved in savings/<log_dir>/ to keep them separate from the main
# runs. To do so, run the following command in the terminal from the root directory of the project.
# ```bash
# $ bash scripts/language/ablation.sh
# ```

device="cuda:0"

# Final norm: output of the final norm of the last layer (standard output of the model)
log_dir="final_norm"

# Runs
session="${log_dir}_llms"
tmux new-session -d -s ${session}
for model in \
    "gemma3 1b 32" \
    "gemma3 4b 4" \
    "gemma3 12b 4" \
    "llama3 1B 1" \
    "llama3 3B 1" \
    "llama3 8B 4" \
    "qwen3 1.7B 4" \
    "qwen3 4B 1" \
    "qwen3 8B 1"
do
    read model_name model_size batch_size <<< ${model}
    run="model_size=${model_size} batch_size=${batch_size} layers=[-1] modules=[rc2norm] device=${device} log_dir=${log_dir}"
    command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
    echo "Running command: ${command}"
    tmux send-keys -t ${session} "${command}" C-m
done
