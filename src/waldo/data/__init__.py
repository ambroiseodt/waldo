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

_language = [
    "MTEB",
    "MTEBConfig",
    "build_mteb",
]

_vision = [
    "AircraftDataset",
    "AircraftDatasetConfig",
    "Cifar10DatasetConfig",
    "Cifar10Dataset",
    "Cifar10CDatasetConfig",
    "Cifar10CDataset",
    "Cifar100DatasetConfig",
    "Cifar100Dataset",
    "Cifar100CDatasetConfig",
    "Cifar100CDataset",
    "DomainNetDataset",
    "DomainNetDatasetConfig",
    "DTDDataset",
    "DTDDatasetConfig",
    "EuroSATDataset",
    "EuroSATDatasetConfig",
    "Flowers102DatasetConfig",
    "Flowers102Dataset",
    "Food101Dataset",
    "Food101DatasetConfig",
    "ImageNetDatasetConfig",
    "ImageNetDataset",
    "ImageNetCDataset",
    "ImageNetCDatasetConfig",
    "IWildCamDataset",
    "IWildCamDatasetConfig",
    "PetDatasetConfig",
    "PetDataset",
    "RxRx1DatasetConfig",
    "RxRx1Dataset",
    "build_loader",
    "build_transform",
    "build_dataset",
    "deterministic_split",
]

__all__ = list(_language) + list(_vision)  # type: ignore


def __dir__():
    return __all__


def __getattr__(name: str) -> Any:
    if name in _language:
        return getattr(importlib.import_module("waldo.data.language"), name)
    if name in _vision:
        return getattr(importlib.import_module("waldo.data.vision"), name)
    raise AttributeError(f"module 'waldo.data' has no attribute {name!r}")
