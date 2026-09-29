# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase ("Finding Waldo in Foundation Models") that probes the internal representations of pretrained transformer foundation models — vision (ViT, DINOv2, DINOv3, AIMv2) and language (Gemma3, Llama3, Qwen3) — **layer by layer and module by module**, then evaluates each representation with linear probing (vision) or MTEB (language). Python 3.12+, package source in `src/waldo`.

## Commands

```bash
# Install (editable). Dev tools only, no GPU needed:
pip install -e ".[dev]"
# Experiments need an NVIDIA GPU + the NVIDIA index (cuml-cu12):
pip install -e ".[exps,figs,dev]" --extra-index-url https://pypi.nvidia.com

# Tests (pytest addopts already enable -v and coverage)
pytest
pytest tests/models/vision                                   # one folder
pytest tests/models/vision/test_dinov2.py::TestDinov2Config  # one class/test

# Lint / format (line length 120; rules in pyproject.toml, incl. required arg/return annotations)
ruff check . && ruff format .

# Run an experiment: OmegaConf CLI = dataclass defaults < YAML file < key=value overrides
python -m apps.vision.eval config=apps/vision/configs/dinov2.yaml dataset_name=cifar10 device=cuda:0
python -m apps.language.eval config=apps/language/configs/qwen3.yaml
```

`scripts/**/*.sh` launch sweeps of these commands inside tmux sessions; `scripts/models/download_models.sh` pre-downloads HF weights (useful behind a proxy). Gated models (Llama, Gemma) need `huggingface-cli login`.

CI (`.github/workflows/testing.yaml`) runs each test folder as a separate job with only `.[dev]` plus pinned extras (`transformers`, `mteb`), so tests must not require a GPU, `cuml`, or network access — model tests use mock blocks from each folder's `conftest.py` rather than real weights.

## Architecture

**Config-dict → typed dataclass factories.** Every component is built through a `build_*` factory (`build_model`, `build_loader`, `build_linear_probing`, `build_mteb`) that takes a plain dict, pops a name key (`model_name`, `dataset_name`) and dispatches via `match`, then converts the rest into the component's `*Config` dataclass with `infra.parser.build_with_type_check` (recursive, type-checked; unknown keys are logged and ignored). Adding a model/dataset means: implement class + `*Config`, add a `case` to the factory, and export it in the package `__init__.py` (the `__init__` files use lazy `importlib` exports listed in `_vision`/`_language`/... lists).

**Module-level probing via hooks** (`src/waldo/models/transformer_block.py`, `models/{vision,language}/base.py`). Every transformer block is decomposed into the 8 canonical modules in `TRANSFORMER_MODULES = [ln1, mha, rc1, ln2, fc1, act, fc2, rc2]`. Each model supplies a `TransformerBlockAdapter` subclass mapping these names to its HF submodules. `get_embeddings` registers forward hooks (or pre-hooks for modules listed by `_get_pre_hooks_modules`) on the selected layers/modules, runs a **single forward pass**, pools each captured tensor immediately (to CPU), and returns `{"block_{i}_{module}": (N, E)}`. Key conventions:
- Pre-norm is assumed; `rc1` is captured as the *input* of `ln2` via a pre-hook.
- For SwiGLU FFNs: `fc1 = gate_proj`, `act = swish(gate)*up` (captured as input of `down_proj` → pre-hook), `fc2 = down_proj`. `process_output` handles special cases (e.g. DINOv2-giant's fused FFN).
- Subclasses implement `_load_model`, `block_adapter`, `blocks`, `final_norm` (+ `n_prefix_tokens` for vision; `pooling`, tokenizer/special tokens/`mteb_model_meta` for LLMs). Models load in eval mode.
- `rc2norm` (`RC2NORM`, not in `TRANSFORMER_MODULES`, so only probed when requested) is the output of the final norm, i.e. the model's standard output (`last_hidden_state`); only valid for the last layer.

**Pooling** (`models/pooling.py`): mean pooling over content tokens everywhere — the `n_prefix_tokens` auxiliary tokens (CLS + registers for DINOv3 and DINOv2 `*-registers` sizes) excluded for vision, padding/BOS/EOS excluded for language — so all layers/modules and models are compared consistently. Vision configs also take `pooling="all"` (CLS concatenated with the patch mean, the DINOv2/v3 linear-probing readout; not for AIMv2), implemented once in `BaseVisionModel.pooling`. The vision apps take `pooling`, `layers` and `modules` (`["all"]` = everything, the default).

**Language models** implement MTEB's `EncoderProtocol` in `BaseLanguageModel` (`encode`, `similarity`, ...) so they can be passed directly to `mteb.evaluate()`.

**Evaluation** (`src/waldo/eval/linear.py`): extracts embeddings for all block/module keys, caches them under the run directory, then fits a StandardScaler + logistic regression per key (cuML on GPU, sklearn otherwise).

**Paths** (`src/waldo/infra/config.py`): all relative to repo root — `datasets/`, `checkpoints/`, `results/`, `savings/` (runs go to `savings/runs/<dataset>/<model>/`), `figures/`. `plots/aggregate.py` collects run outputs into `results/*.csv`; `plots/analysis.py` and `notebooks/` produce figures.

`apps/` holds the experiment entry points (each with an `EvaluationConfig` dataclass + YAML configs); `src/waldo` holds reusable library code.

## Code style

Numpy-style docstrings with `r"""` and a license header block in every module; sections separated by `# ----` banner comments; comment abbreviations N (batch), L (sequence), E (embedding dim). Base classes document the preferred method ordering (init → abstract private → abstract public → abstract properties → private → public → dunder → properties).

## Updating CLAUDE.md
When you change project structure, commands, or conventions, update this file in the same change.