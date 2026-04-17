r"""
Shared fixtures for data vision unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import Dataset

from waldo.data import (
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
    DTDDataset,
    DTDDatasetConfig,
    EuroSATDataset,
    EuroSATDatasetConfig,
    Flowers102Dataset,
    Flowers102DatasetConfig,
    Food101Dataset,
    Food101DatasetConfig,
    ImageNetCDataset,
    ImageNetCDatasetConfig,
    ImageNetDataset,
    ImageNetDatasetConfig,
    PetDataset,
    PetDatasetConfig,
)

# Default parameters
N_SAMPLES_PER_SEVERITY = 10_000
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2


# -----------------------------------------------------------------------------
# Global fixtures
# ------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sample_image() -> Image.Image:
    r"""A random RGB PIL image representative of a real dataset sample."""
    return Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))


@pytest.fixture(scope="session")
def mock_dataset_torchvision_factory(sample_image: Image.Image) -> Callable[..., Dataset[tuple]]:
    r"""A mock dataset for datasets using standard torchvision datasets (FGVCAircraft, Flowers102, Food101)"""

    def _factory(
        config_class: Callable[..., Any],
        dataset_class: Callable[..., Dataset[tuple]],
        patch_target: str,
        n_samples: int = 10,
        **config_kwargs,
    ) -> Dataset[tuple]:
        with patch(patch_target) as mock_torchvision:
            mock_instance = MagicMock()
            mock_instance._image_files = [f"img_{i}.jpg" for i in range(n_samples)]
            mock_instance._labels = list(range(n_samples))
            mock_instance.loader = lambda path: sample_image
            mock_torchvision.return_value = mock_instance
            config = config_class(**config_kwargs)
            dataset = dataset_class(config)
        return dataset

    return _factory


@pytest.fixture
def mock_dataset_cifar_factory() -> Callable[..., Dataset[tuple]]:
    r"""A mock dataset for CIFAR10/100 datasets using data/targets structure."""

    def _factory(
        config_class: Callable[..., Any],
        dataset_class: Callable[..., Dataset[tuple]],
        patch_target: str,
        n_samples: int = 10,
        **config_kwargs,
    ) -> Dataset[tuple]:
        with patch(patch_target) as mock_torchvision:
            mock_instance = MagicMock()
            mock_instance.data = np.random.randint(0, 255, (n_samples, 32, 32, 3), dtype=np.uint8)
            mock_instance.targets = list(range(n_samples))
            mock_torchvision.return_value = mock_instance
            config = config_class(**config_kwargs)
            dataset = dataset_class(config)
        return dataset

    return _factory


@pytest.fixture
def mock_dataset_flowers102_factory() -> Callable[..., Dataset[tuple]]:
    r"""A mock dataset for Oxford Flowers-102 using _images_files/_labels structure."""

    def _factory(
        config_class: Callable[..., Any],
        dataset_class: Callable[..., Dataset[tuple]],
        patch_target: str,
        n_samples: int = 10,
        **config_kwargs,
    ) -> Dataset[tuple]:
        with patch(patch_target) as mock_torchvision:
            mock_instance = MagicMock()
            mock_instance._image_files = [f"img_{i}.jpg" for i in range(n_samples)]
            mock_instance._labels = list(range(n_samples))
            mock_torchvision.return_value = mock_instance
            config = config_class(**config_kwargs)
            dataset = dataset_class(config)
        return dataset

    return _factory


@pytest.fixture
def mock_dataset_pet_factory() -> Callable[..., Dataset[tuple]]:
    r"""A mock dataset for Pet using _images/_labels structure."""

    def _factory(
        config_class: Callable[..., Any],
        dataset_class: Callable[..., Dataset[tuple]],
        patch_target: str,
        n_samples: int = 10,
        **config_kwargs,
    ) -> Dataset[tuple]:
        with patch(patch_target) as mock_torchvision:
            mock_instance = MagicMock()
            mock_instance._images = [f"img_{i}.jpg" for i in range(n_samples)]
            mock_instance._labels = list(range(n_samples))
            mock_torchvision.return_value = mock_instance
            config = config_class(**config_kwargs)
            dataset = dataset_class(config)
        return dataset

    return _factory


@pytest.fixture
def mock_dataset_image_folder_factory() -> Callable[..., Dataset[tuple]]:
    r"""A mock dataset for datasets using torchvision.datasets.ImageFolder (ImageNet, ImageNet-C, DomainNet)"""

    def _factory(
        config_class: Callable[..., Any],
        dataset_class: Callable[..., Dataset[tuple]],
        patch_target: str,
        n_samples: int = 10,
        **config_kwargs,
    ) -> Dataset[tuple]:
        with patch(patch_target) as mock_folder:
            mock_instance = MagicMock()
            mock_instance.imgs = [(f"img_{i}.jpg", i % 10) for i in range(n_samples)]
            mock_folder.return_value = mock_instance
            config = config_class(**config_kwargs)
            dataset = dataset_class(config)
        return dataset

    return _factory


@pytest.fixture
def mock_dataset_numpy_factory(tmp_path: Path) -> Callable[..., Dataset[tuple]]:
    r"""
    A mock dataset for datasets loaded from .npy files on disk (Cifar10-C, Cifar100-C). For fast test, we consider
    10 samples created as random images loaded in a temporary folder with a fixed seed. To test the length of
    the datasets and the severity slicing, 50_000 samples are needed.
    """

    def _factory(
        config_class: Callable[..., Any],
        dataset_class: Callable[..., Dataset[tuple]],
        n_samples: int = 10,
        n_classes: int = 10,
        **config_kwargs,
    ) -> Dataset[tuple]:
        np.random.seed(seed=42)
        images = np.random.randint(0, 255, (n_samples, 32, 32, 3), dtype=np.uint8)
        labels = np.random.randint(0, n_classes, n_samples)
        np.save(tmp_path / f"{config_kwargs.get('corruption_type', 'brightness')}.npy", images)
        np.save(tmp_path / "labels.npy", labels)
        config = config_class(save_dir=tmp_path, **config_kwargs)
        dataset = dataset_class(config)
        return dataset

    return _factory


# -----------------------------------------------------------------------------
# Dataset-specific fixtures
# ------------------------------------------------------------------------------
@pytest.fixture
def mock_aircraft_dataset(mock_dataset_torchvision_factory: Callable[..., Dataset[tuple]]) -> Dataset[tuple]:
    r"""A mock AircraftDataset with 10 samples, no disk I/O."""
    return mock_dataset_torchvision_factory(
        config_class=AircraftDatasetConfig,
        dataset_class=AircraftDataset,
        patch_target="waldo.data.vision.aircraft.torchvision.datasets.FGVCAircraft",
        mode="train",
    )


@pytest.fixture
def mock_cifar10c_dataset(mock_dataset_numpy_factory: Callable[..., Dataset[tuple]]) -> Dataset[tuple]:
    r"""A mock Cifar10CDataset."""
    return mock_dataset_numpy_factory(
        config_class=Cifar10CDatasetConfig,
        dataset_class=Cifar10CDataset,
        n_classes=10,
        corruption_type="brightness",
        corruption_severity=1,
        mode="train",
    )


@pytest.fixture
def mock_cifar10_dataset(mock_dataset_cifar_factory: Callable[..., Dataset[tuple]]) -> Dataset[tuple]:
    r"""A mock Cifar10Dataset with 10 samples, no disk I/O."""
    return mock_dataset_cifar_factory(
        config_class=Cifar10DatasetConfig,
        dataset_class=Cifar10Dataset,
        patch_target="waldo.data.vision.cifar10.torchvision.datasets.CIFAR10",
        mode="train",
    )


@pytest.fixture
def mock_cifar100c_dataset(mock_dataset_numpy_factory: Callable[..., Dataset[tuple]]) -> Dataset[tuple]:
    r"""A mock Cifar100CDataset."""
    return mock_dataset_numpy_factory(
        config_class=Cifar100CDatasetConfig,
        dataset_class=Cifar100CDataset,
        n_classes=100,
        corruption_type="brightness",
        corruption_severity=1,
        mode="train",
    )


@pytest.fixture
def mock_cifar100_dataset(mock_dataset_cifar_factory: Callable[..., Dataset[tuple]]) -> Dataset[tuple]:
    r"""A mock Cifar100Dataset with 10 samples, no disk I/O."""
    return mock_dataset_cifar_factory(
        config_class=Cifar100DatasetConfig,
        dataset_class=Cifar100Dataset,
        patch_target="waldo.data.vision.cifar100.torchvision.datasets.CIFAR100",
        mode="train",
    )


@pytest.fixture
def mock_domainnet_dataset(mock_dataset_image_folder_factory: Callable[..., Dataset[tuple]]) -> Dataset[tuple]:
    r"""A mock DomainNetDataset with 10 samples, no disk I/O."""
    return mock_dataset_image_folder_factory(
        config_class=DomainNetDatasetConfig,
        dataset_class=DomainNetDataset,
        patch_target="waldo.data.vision.domainnet.torchvision.datasets.ImageFolder",
        n_samples=10,
        domain="clipart",
        mode="train",
    )


@pytest.fixture
def mock_dtd_dataset(mock_dataset_torchvision_factory: Callable[..., Dataset[tuple]]) -> Dataset[tuple]:
    r"""A mock DTDDataset with 10 samples, no disk I/O."""
    return mock_dataset_torchvision_factory(
        config_class=DTDDatasetConfig,
        dataset_class=DTDDataset,
        patch_target="waldo.data.vision.dtd.torchvision.datasets.DTD",
        mode="train",
    )


@pytest.fixture
def mock_eurosat_dataset(mock_dataset_image_folder_factory: Callable[..., Dataset[tuple]]) -> Dataset[tuple]:
    r"""A mock EuroSATDataset with 10 samples, no disk I/O."""
    return mock_dataset_image_folder_factory(
        config_class=EuroSATDatasetConfig,
        dataset_class=EuroSATDataset,
        patch_target="waldo.data.vision.eurosat.torchvision.datasets.EuroSAT",
        n_samples=10,
        mode="train",
    )


@pytest.fixture
def mock_flowers102_dataset(mock_dataset_flowers102_factory: Callable[..., Dataset[tuple]]) -> Dataset[tuple]:
    r"""A mock Flowers102Dataset with 10 samples, no disk I/O."""
    return mock_dataset_flowers102_factory(
        config_class=Flowers102DatasetConfig,
        dataset_class=Flowers102Dataset,
        patch_target="waldo.data.vision.flowers102.torchvision.datasets.Flowers102",
        mode="train",
    )


@pytest.fixture
def mock_food101_dataset(mock_dataset_torchvision_factory: Callable[..., Dataset[tuple]]) -> Dataset[tuple]:
    r"""A mock Food101Dataset with 10 samples, no disk I/O."""
    return mock_dataset_torchvision_factory(
        config_class=Food101DatasetConfig,
        dataset_class=Food101Dataset,
        patch_target="waldo.data.vision.food101.torchvision.datasets.Food101",
        mode="train",
    )


@pytest.fixture
def mock_imagenetc_dataset(mock_dataset_image_folder_factory: Callable[..., Dataset[tuple]]) -> Dataset[tuple]:
    r"""A mock ImageNetCDataset with 10 samples, no disk I/O."""
    return mock_dataset_image_folder_factory(
        config_class=ImageNetCDatasetConfig,
        dataset_class=ImageNetCDataset,
        patch_target="waldo.data.vision.imagenet_c.torchvision.datasets.ImageFolder",
        corruption_type="brightness",
        corruption_severity=1,
        mode="train",
    )


@pytest.fixture
def mock_imagenet_dataset(mock_dataset_image_folder_factory: Callable[..., Dataset[tuple]]) -> Dataset[tuple]:
    r"""A mock ImageNetDataset with 10 samples, no disk I/O."""
    return mock_dataset_image_folder_factory(
        config_class=ImageNetDatasetConfig,
        dataset_class=ImageNetDataset,
        patch_target="waldo.data.vision.imagenet.torchvision.datasets.ImageFolder",
        mode="train",
    )


@pytest.fixture
def mock_pet_dataset(mock_dataset_pet_factory: Callable[..., Dataset[tuple]]) -> Dataset[tuple]:
    r"""A mock PetDataset with 10 samples, no disk I/O."""
    return mock_dataset_pet_factory(
        config_class=PetDatasetConfig,
        dataset_class=PetDataset,
        patch_target="waldo.data.vision.pet.torchvision.datasets.OxfordIIITPet",
        mode="train",
    )
