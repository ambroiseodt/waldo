r"""
Codebase to evaluation VFMs and VLMs via linear probing.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from .linear import run_linear_probing

__all__ = [
    "run_linear_probing",
]
