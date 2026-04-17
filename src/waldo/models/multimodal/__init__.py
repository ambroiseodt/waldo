r"""
Codebase to load pretrained VFMs from HuggingFace.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from .clip import Clip, ClipBlockAdapter, ClipConfig
from .siglip2 import Siglip2, Siglip2BlockAdapter, Siglip2Config

__all__ = [
    "Clip",
    "ClipBlockAdapter",
    "ClipConfig",
    "Siglip2",
    "Siglip2BlockAdapter",
    "Siglip2Config",
]
