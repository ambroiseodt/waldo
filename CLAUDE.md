# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- Keep this file short (< 200 lines): only what cannot be derived from the code. Update it in the same change as
the structure, commands or conventions it describes. -->

## Project

Research code for the paper "Probing Transformer Representations Layer by Layer, Module by Module": probe pretrained
vision models (ViT, DINOv2, DINOv3, AIMv2) and LLMs (Gemma3, Llama3, Qwen3) at every layer and every module of the
transformer block, then evaluate each representation with linear probing (vision) or MTEB (language).
Library code in `src/waldo`, experiment entry points and configs in `apps/`, launch scripts in `scripts/`, aggregation
of the runs into `analysis/results/` (`runs/` for the paper, `ablations/`, each with a `summary/` per model) and paper
figures and tables in `analysis/`.

## Commands

```bash
# Tests without GPU (as in CI): CPU PyTorch first, otherwise pip pulls ~4 GB of CUDA libraries
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,test]"
# Experiments (NVIDIA GPU, cuML for GPU linear probing)
pip install -e ".[exps,figs,dev]" --extra-index-url https://pypi.nvidia.com

pytest                                                       # addopts already enable -v and coverage
pytest tests/src/waldo/models/vision/test_dinov2.py::TestDinov2Config  # one class or test
ruff check . && ruff format .                                # line length 120, annotations required

# Experiments: OmegaConf CLI, dataclass defaults < YAML file < key=value overrides
python -m apps.vision.eval config=apps/vision/configs/dinov2.yaml dataset_name=cifar10 device=cuda:0
python -m apps.language.eval config=apps/language/configs/qwen3.yaml layers=[-1] modules=[rc2norm]
```

Gated models (DINOv3, Gemma3, Llama3) need `hf auth login`. Tests mirror `src/waldo/` and `apps/`
(`tests/src/waldo/...`, `tests/apps/...`) and must run without GPU, `cuml` or network access: model tests patch
`from_pretrained` with tiny randomly initialized HuggingFace models (see each folder's `conftest.py`), and
`tests/conftest.py` makes any unpatched `from_pretrained` fail and sets `HF_HUB_OFFLINE=1`.

## Workflow

- Work on `dev`, merge into `main` through a pull request (CI runs on `main`, `dev` and PRs to `main`). Commit or push
  only when asked.
- Commit messages: one capitalized, factual subject line (e.g. `Fix silent errors in model loading`), optional `-`
  bullet list in the body, no emojis.
- IMPORTANT: default options must reproduce the paper results exactly. After any change to models, pooling, hooks or
  linear probing, compare the default outputs with `main` on tiny random models (patch `from_pretrained`): embeddings
  must be identical, byte for byte, unless the change is an intended fix.
- Never delete or overwrite experiment outputs: `savings/` (runs), `analysis/results/` (aggregated CSVs, Git LFS). With
  `overwrite=true`, linear probing deletes the existing results of a run. New, corrected or ablation runs go to another
  directory with the `log_dir` option of the apps (relative to `savings/`, default `runs`, e.g.
  `log_dir=ablations/final_norm`), so that `savings/runs/` and `analysis/` keep reproducing the paper.
- `temp/` is gitignored scratch space (rebuttal notes, check scripts).

## Architecture

- Components are built by `build_*` factories from a plain dict: a name key (`model_name`, `dataset_name`) selects
  the class, the rest becomes its `*Config` dataclass through `infra.parser.build_with_type_check` (unknown keys are
  logged and ignored). A new model or dataset needs a class, a `*Config`, a `case` in the factory and an export in
  the package `__init__.py` (lazy `importlib` exports).
- Each transformer block is decomposed into `TRANSFORMER_MODULES = [ln1, mha, rc1, ln2, fc1, act, fc2, rc2]` by a
  `TransformerBlockAdapter` per model. Hooks capture all requested modules in a single forward pass. Pre-norm is
  assumed: `rc1` is the input of `ln2` (pre-hook). Gated FFNs (SwiGLU, GeGLU): `fc1 = gate_proj`,
  `act = act(gate) * up` (input of `down_proj`, pre-hook), `fc2 = down_proj`.
- Vision models pool inside the hooks (`BaseVisionModel.pooling`, `pooling="content"` or `"all"` = CLS and patch
  mean). LLMs keep full sequences and pool with the attention mask, excluding BOS/EOS/padding, and implement MTEB's
  `EncoderProtocol`.

## Code style

- Numpy-style docstrings with `r"""`, license header in every module, sections separated by `# ----` banners,
  abbreviations N (batch), L (sequence), E (embedding dimension) in comments.
- Method order in classes: `__init__`, abstract private, abstract public, abstract properties, private, public,
  dunder, properties.
- Configs validate their fields in `__post_init__` and raise `ValueError` with the valid options.
