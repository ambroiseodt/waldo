r"""
Dataset and dataloader instantiation factory for vision benchmarks.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from ...infra import build_with_type_check


# ------------------------------------------------------------------------------
#   Dataset
# ------------------------------------------------------------------------------
def deterministic_split(data: np.ndarray, train_size: float = 0.8, train: bool = True) -> np.ndarray:
    r"""
    Create a deterministic split of the original data.

    Parameters
    ----------
    data: np.ndarray
        The data to be split.
    train_size: float
        The proportion of the data to be used for training.
    train: bool
        Whether to return the training set or the test set.

    Returns
    -------
    indices: np.ndarray
        The indices of the data to be used for training or testing.
    """
    n_samples = len(data)
    n_train = int(train_size * n_samples)

    # Apply a deterministic random permutation of the data
    st0 = np.random.get_state()
    np.random.seed(42)
    indices = np.random.permutation(range(n_samples))
    np.random.set_state(st0)

    # Recover the desired subset of indices
    if train:
        indices = indices[:n_train]
    else:
        indices = indices[n_train:]

    return np.sort(indices)


def build_dataset(config: dict[str, Any]) -> tuple[Dataset, int]:
    r"""
    Initialize the dataset based on the specified implementation given in the config file.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration details.

    Returns
    -------
    dataset, n_classes: tuple[Dataset, int]
        An instance of the specified dataset implementation and the number of classes in the dataset.
    """

    dataset_name = config.pop("dataset_name", "cifar10")
    match dataset_name.lower():
        case "aircraft":
            from waldo.data.vision.aircraft import AircraftDataset, AircraftDatasetConfig

            dataset_type = AircraftDataset
            config_obj = build_with_type_check(AircraftDatasetConfig, config)

        case x if "cifar10_c" in x:
            from waldo.data.vision.cifar10_c import Cifar10CDataset, Cifar10CDatasetConfig

            dataset_type = Cifar10CDataset

            # Dataset name format should be cifar10_c-corruption-brightness-severity-1
            substring = dataset_name.split("cifar10_c", 1)[-1]
            substring = substring.split("-corruption-", 1)[-1]
            config["corruption_type"], config["corruption_severity"] = substring.split("-severity-", 1)
            config_obj = build_with_type_check(Cifar10CDatasetConfig, config)

        case "cifar10":
            from waldo.data.vision.cifar10 import Cifar10Dataset, Cifar10DatasetConfig

            dataset_type = Cifar10Dataset
            config_obj = build_with_type_check(Cifar10DatasetConfig, config)

        case "cifar100":
            from waldo.data.vision.cifar100 import Cifar100Dataset, Cifar100DatasetConfig

            dataset_type = Cifar100Dataset
            config_obj = build_with_type_check(Cifar100DatasetConfig, config)

        case x if "domainnet" in x:
            from waldo.data.vision.domainnet import DomainNetDataset, DomainNetDatasetConfig

            dataset_type = DomainNetDataset

            # Dataset name format should be domainnet-clipart
            config["domain"] = dataset_name.split("domainnet-", 1)[-1]
            config_obj = build_with_type_check(DomainNetDatasetConfig, config)

        case "dtd":
            from waldo.data.vision.dtd import DTDDataset, DTDDatasetConfig

            dataset_type = DTDDataset
            config_obj = build_with_type_check(DTDDatasetConfig, config)

        case "eurosat":
            from waldo.data.vision.eurosat import EuroSATDataset, EuroSATDatasetConfig

            dataset_type = EuroSATDataset
            config_obj = build_with_type_check(EuroSATDatasetConfig, config)

        case "flowers102":
            from waldo.data.vision.flowers102 import Flowers102Dataset, Flowers102DatasetConfig

            dataset_type = Flowers102Dataset
            config_obj = build_with_type_check(Flowers102DatasetConfig, config)

        case "food101":
            from waldo.data.vision.food101 import Food101Dataset, Food101DatasetConfig

            dataset_type = Food101Dataset
            config_obj = build_with_type_check(Food101DatasetConfig, config)

        case "iwildcam":
            from waldo.data.vision.iwildcam import IWildCamDataset, IWildCamDatasetConfig

            dataset_type = IWildCamDataset
            config_obj = build_with_type_check(IWildCamDatasetConfig, config)

        case "pet":
            from waldo.data.vision.pet import PetDataset, PetDatasetConfig

            dataset_type = PetDataset
            config_obj = build_with_type_check(PetDatasetConfig, config)

        case "rxrx1":
            from waldo.data.vision.rxrx1 import RxRx1Dataset, RxRx1DatasetConfig

            dataset_type = RxRx1Dataset
            config_obj = build_with_type_check(RxRx1DatasetConfig, config)

        case _:
            raise ValueError(f"Dataset name {dataset_name} not found.")

    # Build dataset
    dataset = dataset_type(config_obj)

    return dataset, dataset.n_classes


# ------------------------------------------------------------------------------
# Data preprocessing
# ------------------------------------------------------------------------------
def build_transform(resolution: int, transform_type: str) -> Compose:
    r"""
    Image preprocessing following [1]_.

    Parameters
    ----------
    resolution: int
        Resolution of the image after resizing.
    transform_type: str
        Type of transform to apply. Options are "train", "val", "test" and "static".
        It should be set as static for linear probing on static frozen features.

    Returns
    -------
    transform: Compose
        Composition of the transforms to apply.

    Notes
    -----
    For linear probing on static frozen features, data augmentation might be detrimental since each training sample is
    seen only once by the classifier. In this case, we follow the standard practice (see [2]_) by resizing and applying
    center crop and normalization with ImageNet statistics.

    References
    ----------
    .. [1] A. Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
           In ICLR 2021

    .. [2] A. Radford et al. Learning Transferable Visual Models From Natural Language Supervision. In ICML 2021
    """
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    match transform_type.lower():
        case "train":
            transform = Compose(
                [
                    RandomResizedCrop(resolution),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    normalize,
                ]
            )
        case "val" | "test" | "static":
            transform = Compose(
                [
                    Resize(resolution),
                    CenterCrop(resolution),
                    ToTensor(),
                    normalize,
                ]
            )
        case _:
            raise ValueError(f"Type {transform_type} not found. Options are 'train', 'val', 'test' and 'static'.")

    return transform


# ------------------------------------------------------------------------------
#   Dataloader
# ------------------------------------------------------------------------------
def _worker_init_fn(worker_id: int) -> None:
    r"""
    Fix the number of threads used by each worker to 1 to avoid unnecessary parallelization.

    Parameters
    ----------
    worker_id: int
        The worker ID.
    """
    import torch

    torch.set_num_threads(1)


def build_loader(
    config: dict[str, Any],
    n_workers: int = 8,
    return_n_classes: bool = False,
) -> DataLoader | tuple[DataLoader, int]:
    r"""
    Build dataloader with optimized parallelism settings.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration details.
    n_workers: int, default=8
        The number of workers to use for data loading.
    return_n_classes: bool, default=False
        Whether to return the number of classes in the dataset.

    Returns
    -------
    loader: DataLoader | tuple[DataLoader, int]
        An instance of the specified dataloader implementation.

    Notes
    -----
    Loading data can be slow if the n_workers variable is not properly set. The default value is set to 8 based on
    empirical benchmarking on our datasets. Despite light augmentations, multiple workers improve throughput due to
    I/O parallelism when reading from disk. OMP_NUM_THREADS is set to 1 per worker via worker_init_fn to prevent thread
    contention: if n_workers > 0, each worker spawns its own OpenMP threads, leading to n_workers x OMP_NUM_THREADS
    competing threads. Restricting workers to a single thread avoids this while leaving the main process thread budget
    untouched for model training. This follows the protocol of https://github.com/facebookresearch/lingua.
    """

    # Argument parsing
    batch_size = config.pop("batch_size", 128)
    resolution = config.pop("resolution", 224)
    transform_type = config.pop("transform_type", "train")

    # Set data transform
    config = config | dict(transform=build_transform(resolution=resolution, transform_type=transform_type))

    # Create dataset
    dataset, n_classes = build_dataset(config)

    # Create dataloader
    shuffle = config["mode"] == "train"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=_worker_init_fn,
        num_workers=n_workers,
        pin_memory=True,
    )

    # Recover number of classes
    if return_n_classes:
        return loader, n_classes

    return loader
