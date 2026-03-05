r"""
Codebase to preprocess vision and languages benchmarks.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from .language import MTEB, MTEBConfig, build_mteb
from .vision import (
    AircraftDataset,
    AircraftDatasetConfig,
    Cifar10CDataset,
    Cifar10CDatasetConfig,
    Cifar10Dataset,
    Cifar10DatasetConfig,
    Cifar100CDataset,
    Cifar100CDatasetConfig,
    Cifar100Dataset,
    Cifar100DatasetConfig,
    DomainNetDataset,
    DomainNetDatasetConfig,
    Flowers102Dataset,
    Flowers102DatasetConfig,
    Food101Dataset,
    Food101DatasetConfig,
    ImageNetCDataset,
    ImageNetCDatasetConfig,
    ImageNetDataset,
    ImageNetDatasetConfig,
    OxfordIIITPetDataset,
    OxfordIIITPetDatasetConfig,
    build_loader,
)

__all__ = [
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
    "ImageNetDatasetConfig",
    "ImageNetDataset",
    "ImageNetCDataset",
    "ImageNetCDatasetConfig",
    "Flowers102DatasetConfig",
    "Flowers102Dataset",
    "Food101Dataset",
    "Food101DatasetConfig",
    "MTEB",
    "MTEBConfig",
    "OxfordIIITPetDatasetConfig",
    "OxfordIIITPetDataset",
    "build_mteb",
    "build_loader",
]
