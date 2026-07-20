# Finding Waldo in Foundation Models

[![Tests](https://github.com/ambroiseodt/waldo/actions/workflows/testing.yaml/badge.svg)](https://github.com/ambroiseodt/waldo/actions)

## Getting started
The code runs Python 3.12+. Here are some installation instructions:
Install [miniforge](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). Follow the instruction online, most likely you will execute the following commands:
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash ~/Miniforge3-latest-Linux-x86_64.sh
source ~/.bashrc
```
Install Python in a new conda environment (be mindful to install a Python version compatible with Pytorch):
```bash
conda create -n myenv python==3.12
conda activate myenv
```
Install the repository (the ``pretrained" dependencies are optional but allow for a faster download of weights):
```bash
git clone <repo url>
cd <repo path>
pip install -e ".[exps]"
```
To install the development and visualization dependencies, you can swap the previous command for the following one:
```bash
pip install -e ".[exps,dev,figs]"
```

#### Accelerate specific instructions
The accelerate package can be used to download and distribute models from the HuggingFace Transformers library. After installing it, one needs to configure it. Follow the instruction online [configure-accelerate](https://huggingface.co/docs/accelerate/en/basic_tutorials/install), most likely you will execute the following command and answer the questions prompted to you:
```bash
accelerate config
```

> [!NOTE]
> Libraries such as [transformers](https://github.com/huggingface/transformers), [OpenClip](https://github.com/mlfoundations/open_clip) or [MTEB](https://github.com/embeddings-benchmark/mteb) are subject to frequent changes which might impact the behavior of the codebase. We invite users to refer to the official documentation of those libraries if they encounter installation or dependency issues.

### Using HuggingFace pretrained models
Some models are gated, e.g., the Llama and Gemma ones. Users should request the access and login to the huggingface hub to use them in the scripts. See https://huggingface.co/docs/hub/en/models-gated for more information.

## Installation

### Why `pip install -e`?

The `-e` flag installs the package in **editable mode** — changes to the source code in
`src/` are immediately reflected without reinstalling. This is the recommended way to
install the package for both running experiments and contributing to the codebase.

### Dependency groups

| Group | Description |
|-------|-------------|
| _(none)_ | Core dependencies (`torch`, `numpy`, etc.) |
| `experiments` | Everything needed to run experiments. Requires an NVIDIA GPU and CUDA. |
| `figures` | Visualization tools for plotting results. |
| `dev` | Development tools (`pytest`, `ruff`) for contributors. |

### Common setups

```bash
# Run all experiments (requires NVIDIA GPU + CUDA)
pip install -e ".[exps]" --extra-index-url https://pypi.nvidia.com

# Run experiments and plot results
pip install -e ".[exps,figs]" --extra-index-url https://pypi.nvidia.com

# Contribute to the codebase (no GPU needed)
pip install -e ".[dev]"

# Full setup
pip install -e ".[exps,figs,dev]" --extra-index-url https://pypi.nvidia.com
```

> [!NOTE]
> The `experiments` group includes `cuml-cu12` for GPU-accelerated linear probing,
> which requires an NVIDIA GPU, CUDA, and the NVIDIA PyPI index. Always append
> `--extra-index-url https://pypi.nvidia.com` when installing `experiments`. See [cuml](https://github.com/rapidsai/cuml) for details.

## Testing

Install the development dependencies:
```bash
pip install -e ".[dev]"
```

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=term-missing
```

To run the tests of a specific folder (or a file) located in tests/my_path, run
```bash
pytest tests/my_path
```

## Problem with proxy
If issue with huggingface model download, pre-download them with scripts/download_models.py and set load_offline=True in build_model
Update: the previous issue seems to be resolved (alone, I don't know why it didn't work or why it works now). In all cases, you can download all models before-hand using download_models.sh

## Acknowledgements

## Contact
If you have any questions, feel free to reach out at [```ambroiseodonnattechnologie@gmail.com```](mailto:ambroiseodonnattechnologie@gmail.com).

## Citation
