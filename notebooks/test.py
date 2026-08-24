# from waldo.data import build_mteb

# mteb_config = {
#     "benchmark_name": "Multilingual",
#     "task_type": "Classification",
#     "max_n_docs": 8192,
#     "batch_size": 8,
# }
# mteb_benchmark = build_mteb(config=mteb_config)
# print(mteb_benchmark.tasks, mteb_benchmark.domains, mteb_benchmark.languages)
# print(mteb_benchmark.n_tasks, len(mteb_benchmark.domains), len(mteb_benchmark.languages))
import torch
from transformers import AutoConfig, AutoModelForCausalLM

for model_id in [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "google/gemma-3-1b-pt",
    "google/gemma-3-4b-pt",
    "google/gemma-3-12b-pt",
]:
    print(model_id)
    # Fetch architecture specs from config
    config = AutoConfig.from_pretrained(model_id)

    # Load model shell (meta device prevents downloading large weights into RAM)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"--- Model Specs: {model_id} ---")
    print(f"Embedding Dim (Hidden Size) : {config.hidden_size}")
    print(f"FFN Dim (Intermediate Size): {config.intermediate_size}")
    print(f"Number of Layers            : {config.num_hidden_layers}")
    print(f"Number of Attention Heads   : {config.num_attention_heads}")
    print(f"Total Parameters            : {total_params:,} ({total_params / 1e9:.2f}B)")
