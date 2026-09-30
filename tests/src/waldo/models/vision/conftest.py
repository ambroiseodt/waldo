r"""
Shared fixtures for vision foundation models unit tests.

Models are built from tiny randomly initialized HuggingFace models with the architecture of the pretrained ones,
so that tests run without network access or pretrained weights.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import (
    Aimv2VisionConfig,
    Aimv2VisionModel,
    Dinov2Config,
    Dinov2Model,
    Dinov2WithRegistersConfig,
    Dinov2WithRegistersModel,
    DINOv3ViTConfig,
    DINOv3ViTModel,
    PretrainedConfig,
    PreTrainedModel,
    ViTConfig,
    ViTModel,
)

from waldo.models import BaseVisionModel, build_model

# Default parameters
BATCH_SIZE = 2
IMAGE_SIZE = 32
PATCH_SIZE = 8
N_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
N_LAYERS = 2
HIDDEN_SIZE = 32
TINY = dict(hidden_size=HIDDEN_SIZE, num_hidden_layers=N_LAYERS, num_attention_heads=4, intermediate_size=64)
TINY_IMAGE = dict(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE)

# Type of the make_vision_model fixture
MakeVisionModel = Callable[..., tuple[BaseVisionModel, PreTrainedModel]]


@dataclass
class VisionSpec:
    r"""Tiny HuggingFace model standing for a pretrained model, and the waldo config that loads it."""

    hf_class: type[PreTrainedModel]
    hf_config: PretrainedConfig
    config: dict[str, Any]
    n_prefix_tokens: int
    gated: bool = False


VISION_SPECS = {
    "vit": VisionSpec(ViTModel, ViTConfig(**TINY, **TINY_IMAGE), {"model_name": "vit"}, n_prefix_tokens=1),
    "dinov2": VisionSpec(
        Dinov2Model, Dinov2Config(**TINY, **TINY_IMAGE, layerscale_value=0.5), {"model_name": "dinov2"}, 1
    ),
    "dinov2-swiglu": VisionSpec(
        Dinov2Model,
        Dinov2Config(**TINY, **TINY_IMAGE, layerscale_value=0.5, use_swiglu_ffn=True),
        {"model_name": "dinov2", "model_size": "giant"},
        n_prefix_tokens=1,
        gated=True,
    ),
    "dinov2-registers": VisionSpec(
        Dinov2WithRegistersModel,
        Dinov2WithRegistersConfig(**TINY, **TINY_IMAGE, layerscale_value=0.5, num_register_tokens=4),
        {"model_name": "dinov2", "model_size": "base-registers"},
        n_prefix_tokens=5,
    ),
    "dinov3": VisionSpec(
        DINOv3ViTModel,
        DINOv3ViTConfig(**TINY, **TINY_IMAGE, layerscale_value=0.5, num_register_tokens=4),
        {"model_name": "dinov3"},
        n_prefix_tokens=5,
    ),
    "dinov3-gated": VisionSpec(
        DINOv3ViTModel,
        DINOv3ViTConfig(**TINY, **TINY_IMAGE, layerscale_value=0.5, num_register_tokens=4, use_gated_mlp=True),
        {"model_name": "dinov3", "model_size": "vit7b16"},
        n_prefix_tokens=5,
        gated=True,
    ),
    "aimv2": VisionSpec(
        Aimv2VisionModel, Aimv2VisionConfig(**TINY, **TINY_IMAGE), {"model_name": "aimv2"}, 0, gated=True
    ),
}


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------
@pytest.fixture
def make_vision_model() -> MakeVisionModel:
    r"""Build a waldo vision model on top of the tiny HuggingFace model of a spec, and return both."""

    def _factory(name: str, **config: Any) -> tuple[BaseVisionModel, PreTrainedModel]:
        spec = VISION_SPECS[name]
        torch.manual_seed(0)
        hf_model = spec.hf_class(spec.hf_config).eval()
        with patch.object(spec.hf_class, "from_pretrained", return_value=hf_model):
            model = build_model({**spec.config, "device": "cpu", **config})
        return model, hf_model

    return _factory


@pytest.fixture(params=list(VISION_SPECS))
def vision_spec(request: pytest.FixtureRequest) -> str:
    r"""Name of each vision spec, to run a test on all architectures."""
    return request.param


@pytest.fixture
def images() -> torch.Tensor:
    r"""A batch of random images."""
    return torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, generator=torch.Generator().manual_seed(0))
