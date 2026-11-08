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
from .domainnet import DomainNetDataset, DomainNetDatasetConfig
from .dtd import DTDDataset, DTDDatasetConfig
from .eurosat import EuroSATDataset, EuroSATDatasetConfig
from .factory import build_dataset, build_loader, build_transform, deterministic_split
from .flowers102 import Flowers102Dataset, Flowers102DatasetConfig
from .food101 import Food101Dataset, Food101DatasetConfig
from .iwildcam import IWildCamDataset, IWildCamDatasetConfig
from .pet import PetDataset, PetDatasetConfig
from .rxrx1 import RxRx1Dataset, RxRx1DatasetConfig

__all__ = [
    "AircraftDataset",
    "AircraftDatasetConfig",
    "Cifar10DatasetConfig",
    "Cifar10Dataset",
    "Cifar10CDatasetConfig",
    "Cifar10CDataset",
    "Cifar100DatasetConfig",
    "Cifar100Dataset",
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
