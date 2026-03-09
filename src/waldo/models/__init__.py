r"""
Codebase to load pretrained LVMs, VLMs and LLMs from HuggingFace.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from .vision import ViT, ViTConfig

__all__ = ["ViT", "ViTConfig"]
