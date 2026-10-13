#!/usr/bin/bash

# This file is useful to run the ablations of LLMs on the MTEB benchmark. It will create a
# dedicated tmux session on the specified device, where the models are evaluated one after
# the other. The runs are saved in savings/ablations/<ablation>/ to keep them separate from the main
# runs. To do so, run the following command in the terminal from the root directory of the project.
# ```bash
# $ bash scripts/language/ablation.sh
# ```

# Final norm
ablation="final_norm"
log_dir="ablations/${ablation}"
batch_size=1
device="cuda:0"

# Runs
session="${ablation}_mteb"
tmux new-session -d -s ${session}
for config in \
    "gemma3 1b" \
    "gemma3 4b" \
    "gemma3 12b" \
    "llama3 1B" \
    "llama3 3B" \
    "llama3 8B" \
    "qwen3 1.7B" \
    "qwen3 4B" \
    "qwen3 8B" 
do
    read model_name model_size <<< ${config}
    run="log_dir=${log_dir} model_size=${model_size} batch_size=${batch_size} layers=[-1] modules=[rc2norm] device=${device}"
    command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
    echo "Running command: ${command}"
    tmux send-keys -t ${session} "${command}" C-m
done

# Gemma4-E2B (default probing: all layers and modules)
ablation="gemma4"
log_dir="ablations/${ablation}"
model_name="gemma4"
model_size="E2B"
batch_size=1
device="cuda:0"

# Runs
session="${ablation}_mteb"
tmux new-session -d -s ${session}
run="log_dir=${log_dir} model_size=${model_size} batch_size=${batch_size} device=${device}"
command="python -m apps.language.eval config=apps/language/configs/${model_name}.yaml ${run}"
echo "Running command: ${command}"
tmux send-keys -t ${session} "${command}" C-m
