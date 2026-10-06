r"""
Shared fixtures for experiment entry points unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from waldo.infra import ROOT_DIR, build_with_type_check
from waldo.models import (
    Aimv2Config,
    ClipConfig,
    Dinov2Config,
    Dinov3Config,
    Gemma3Config,
    Llama3Config,
    Qwen3Config,
    SmolLM2Config,
    ViTConfig,
)

CONFIG_DIR = ROOT_DIR / "apps"
MODEL_CONFIGS = {
    "vit": ViTConfig,
    "dinov2": Dinov2Config,
    "dinov3": Dinov3Config,
    "aimv2": Aimv2Config,
    "clip": ClipConfig,
    "gemma3": Gemma3Config,
    "llama3": Llama3Config,
    "qwen3": Qwen3Config,
    "smollm2": SmolLM2Config,
}


def load_config(evaluation_config: type, path: Path, **overrides: Any) -> Any:
    r"""Merge the defaults of an app, a YAML configuration file and overrides, as the command line interface does."""
    merged = OmegaConf.merge(OmegaConf.structured(evaluation_config()), OmegaConf.load(path), overrides)
    return OmegaConf.to_object(merged)


def check_model_config(config: Any, keys: list[str]) -> None:
    r"""Build the model configuration of an app config, which raises a ValueError if a value is invalid."""
    model_config = MODEL_CONFIGS[config.model_name]
    build_with_type_check(model_config, {key: getattr(config, key) for key in keys})
