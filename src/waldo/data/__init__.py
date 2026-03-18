r"""
Codebase to preprocess vision and languages benchmarks.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import importlib
from typing import Any

_vision = [
    "AircraftDataset",
    "AircraftDatasetConfig",
    "Cifar10Dataset",
    "Cifar10DatasetConfig",
    "Cifar10CDataset",
    "Cifar10CDatasetConfig",
    "Cifar100Dataset",
    "Cifar100DatasetConfig",
    "Cifar100CDataset",
    "Cifar100CDatasetConfig",
    "DomainNetDataset",
    "DomainNetDatasetConfig",
    "DTDDataset",
    "DTDDatasetConfig",
    "EuroSATDataset",
    "EuroSATDatasetConfig",
    "Flowers102Dataset",
    "Flowers102DatasetConfig",
    "Food101Dataset",
    "Food101DatasetConfig",
    "ImageNetDataset",
    "ImageNetDatasetConfig",
    "ImageNetCDataset",
    "ImageNetCDatasetConfig",
    "PetDataset",
    "PetDatasetConfig",
    "build_loader",
    "build_transform",
]

_language = [
    "MTEB",
    "MTEBConfig",
    "build_mteb",
]

__all__ = list(_vision) + list(_language)  # type: ignore


def __dir__():
    return __all__


def __getattr__(name: str) -> Any:
    if name in _vision:
        return getattr(importlib.import_module("waldo.data.vision"), name)
    if name in _language:
        return getattr(importlib.import_module("waldo.data.language"), name)
    raise AttributeError(f"module 'waldo.data' has no attribute {name!r}")
