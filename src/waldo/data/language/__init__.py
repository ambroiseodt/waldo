r"""
Codebase to preprocess language benchmarks.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from .mteb import MTEB, MTEBConfig
from .utils import build_mteb

__all__ = [
    "MTEB",
    "MTEBConfig",
    "build_mteb",
]
