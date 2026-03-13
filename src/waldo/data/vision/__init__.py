r"""
Codebase to preprocess vision benchmarks.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from .aircraft import AircraftDataset, AircraftDatasetConfig
from .cifar10 import Cifar10Dataset, Cifar10DatasetConfig
from .cifar10_c import Cifar10CDataset, Cifar10CDatasetConfig
from .cifar100 import Cifar100Dataset, Cifar100DatasetConfig
from .cifar100_c import Cifar100CDataset, Cifar100CDatasetConfig
from .domainnet import DomainNetDataset, DomainNetDatasetConfig
from .flowers102 import Flowers102Dataset, Flowers102DatasetConfig
from .food101 import Food101Dataset, Food101DatasetConfig
from .imagenet import ImageNetDataset, ImageNetDatasetConfig
from .imagenet_c import ImageNetCDataset, ImageNetCDatasetConfig
from .pet import OxfordIIITPetDataset, OxfordIIITPetDatasetConfig
from .utils import build_loader, build_transform

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
    "OxfordIIITPetDatasetConfig",
    "OxfordIIITPetDataset",
    "build_loader",
    "build_transform",
]
