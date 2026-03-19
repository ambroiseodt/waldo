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
    "Aimv2Config",
    "Dinov2",
    "Dinov2Config",
    "Dinov3",
    "Dinov3Config",
    "ViT",
    "ViTConfig",
]

_language = []

_multimodal = []

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
    raise AttributeError(f"module 'waldo.data' has no attribute {name!r}")
