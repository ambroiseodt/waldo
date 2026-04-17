r"""
Codebase to load pretrained VFMs, VLMs and LLMs from HuggingFace.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import importlib
from typing import Any

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

__all__ = list(_vision) + list(_language) + list(_multimodal)  # type: ignore


def __dir__():
    return __all__


def __getattr__(name: str) -> Any:
    if name in _vision:
        return getattr(importlib.import_module("waldo.models.vision"), name)
    if name in _language:
        return getattr(importlib.import_module("waldo.models.language"), name)
    if name in _multimodal:
        return getattr(importlib.import_module("waldo.models.multimodal"), name)
    raise AttributeError(f"module 'waldo.models' has no attribute {name!r}")
