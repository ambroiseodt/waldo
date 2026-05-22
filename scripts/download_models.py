r"""
Download HuggingFace models to the default cache. This file is useful when the from_pretrained
method hangs due to proxy issues. It downloads models file-by-file using a method that works through proxies.

Usage
-----
Add the desired models and files to the MODELS and FILES lists, then run from the root of the repository:
```python
python -m scripts.download_models
```

See also
--------
build_model in waldo/models/utils.py

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import os

from huggingface_hub import constants, hf_hub_download

# Increase timeout for large files
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
constants.DEFAULT_ETAG_TIMEOUT = 300  # Internal huggingface_hub timeout

# Models list
MODELS = [
    # "google/gemma-3-1b-pt",
    "google/gemma-3-4b-pt",
    # "google/gemma-3-12b-pt",
    # "google/gemma-3-27b-pt",
    # "google/gemma-4-E2B",
    # "google/gemma-4-E4B",
    # "google/gemma-4-31B",
    # "meta-llama/Llama-3.2-1B",
    # "meta-llama/Llama-3.2-3B",
    # "meta-llama/Llama-3.1-8B",
    # "meta-llama/Llama-3.1-70B",
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-1.7B",
    # "Qwen/Qwen3-4B",
    # "Qwen/Qwen3-8B",
    # "Qwen/Qwen3-14B",
    # "Qwen/Qwen3-32B",
]

# File lists
FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]

# Load models file by file
for model in MODELS:
    print(f"\n{model}")
    for file in FILES:
        print(f"  {file}...", end=" ", flush=True)
        try:
            hf_hub_download(repo_id=model, filename=file)
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
