r"""
Codebase to study pretrained VFMs, VLMs and LLMs from HuggingFace.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import importlib
from typing import Any

_factory = [
    "build_model",
]

_language = [
    "Gemma3",
    "Gemma3BlockAdapter",
    "Gemma3Config",
    "Gemma4",
    "Gemma4BlockAdapter",
    "Gemma4Config",
    "Llama3",
    "Llama3BlockAdapter",
    "Llama3Config",
    "Qwen3",
    "Qwen3BlockAdapter",
    "Qwen3Config",
]

_multimodal = [
    "Clip",
    "ClipBlockAdapter",
    "ClipConfig",
    "Siglip2",
    "Siglip2BlockAdapter",
    "Siglip2Config",
]

_pooling = [
    "mean_pooling",
]


_transformer_block = [
    "SwiGLUGateProjection",
    "TransformerBlockAdapter",
    "TRANSFORMER_MODULES",
]


_vision = [
    "Aimv2",
    "Aimv2BlockAdapter",
    "Aimv2Config",
    "Dinov2",
    "Dinov2BlockAdapter",
    "Dinov2Config",
    "Dinov3",
    "Dinov3BlockAdapter",
    "Dinov3Config",
    "ViT",
    "ViTBlockAdapter",
    "ViTConfig",
]


__all__ = (
    list(_factory) + list(_language) + list(_multimodal) + list(_pooling) + list(_transformer_block) + list(_vision)
)  # type: ignore


def __dir__():
    return __all__


def __getattr__(name: str) -> Any:
    if name in _factory:
        return getattr(importlib.import_module("waldo.models.factory"), name)
    if name in _language:
        return getattr(importlib.import_module("waldo.models.language"), name)
    if name in _multimodal:
        return getattr(importlib.import_module("waldo.models.multimodal"), name)
    if name in _pooling:
        return getattr(importlib.import_module("waldo.models.pooling"), name)
    if name in _transformer_block:
        return getattr(importlib.import_module("waldo.models.transformer_block"), name)
    if name in _vision:
        return getattr(importlib.import_module("waldo.models.vision"), name)
    raise AttributeError(f"module 'waldo.models' has no attribute {name!r}")
