r"""
Codebase for infrastructure and system-level operations.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from .config import DATASET_DIR, DEVICE, FIGURE_DIR, MODEL_DIR, RESULT_DIR, ROOT_DIR, SAVING_DIR, set_seed
from .parser import build_with_type_check, json_serializable, load_jsonl_to_numpy

__all__ = [
    "DATASET_DIR",
    "DEVICE",
    "FIGURE_DIR",
    "MODEL_DIR",
    "RESULT_DIR",
    "ROOT_DIR",
    "SAVING_DIR",
    "build_with_type_check",
    "json_serializable",
    "load_jsonl_to_numpy",
    "set_seed",
]
