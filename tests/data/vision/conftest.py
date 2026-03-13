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
    DomainNetDataset,
    DomainNetDatasetConfig,
    ImageNetDataset,
    ImageNetDatasetConfig,
)

# -----------------------------------------------------------------------------
# Global fixtures
# ------------------------------------------------------------------------------


@pytest.fixture
def sample_image() -> Image.Image:
    r"""A small RGB PIL image representative of a real dataset sample."""
    return Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))


@pytest.fixture
def mock_dataset_torchvision_factory(sample_image: Image.Image) -> Callable[..., Dataset[tuple]]:
    """For datasets using standard torchvision datasets (FGVCAircraft, CIFAR10, etc.)"""

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
def mock_dataset_image_folder_factory() -> Callable[..., Dataset[tuple]]:
    """For datasets using torchvision.datasets.ImageFolder (ImageNet, DomainNet, etc.)"""

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
    """For datasets loaded from .npy files on disk (Cifar10-C, Cifar100-C, etc.)"""

    def _factory(
        config_class: Callable[..., Any],
        dataset_class: Callable[..., Dataset[tuple]],
        n_samples: int = 10,
        n_classes: int = 10,
        **config_kwargs,
    ) -> Dataset[tuple]:
        # Write synthetic .npy files to tmp_path
        images = np.random.randint(0, 255, (n_samples, 32, 32, 3), dtype=np.uint8)
        labels = np.random.randint(0, n_classes, n_samples)
        np.save(tmp_path / f"{config_kwargs.get('corruption_type', 'brightness')}.npy", images)
        np.save(tmp_path / "labels.npy", labels)
        config = config_class(save_dir=tmp_path, **config_kwargs)
        dataset = dataset_class(config)
        return dataset

    return _factory


# -----------------------------------------------------------------------------
# Dataset-specific type
# ------------------------------------------------------------------------------


@pytest.fixture
def mock_aircraft_dataset(mock_dataset_torchvision_factory: Callable[..., Dataset[tuple]]) -> Dataset[tuple]:
    """A mock AircraftDataset with 10 samples, no disk I/O."""
    return mock_dataset_torchvision_factory(
        config_class=AircraftDatasetConfig,
        dataset_class=AircraftDataset,
        patch_target="waldo.data.vision.aircraft.torchvision.datasets.FGVCAircraft",
        mode="train",
    )


@pytest.fixture
def mock_cifar10c_dataset() -> Dataset:
    r"""Here 10_000 samples are needed to slice by severity."""
    return mock_dataset_numpy_factory(
        Cifar10CDataset,
        Cifar10CDatasetConfig,
        n_samples=10_000,
        n_classes=10,
        corruption_type="brightness",
        corruption_severity=1,
        mode="train",
    )


@pytest.fixture
def mock_cifar10_dataset() -> Dataset:
    return mock_dataset_image_folder_factory(
        Cifar10Dataset,
        Cifar10DatasetConfig,
        patch_target="waldo.data.vision.cifar10.torchvision.datasets.CIFAR10",
        mode="train",
    )


@pytest.fixture
def mock_domainnet_dataset(sample_image: Image.Image) -> Dataset:
    with patch("waldo.data.vision.domainnet.Image.open", return_value=sample_image):
        return mock_dataset_image_folder_factory(
            DomainNetDataset,
            DomainNetDatasetConfig,
            patch_target="waldo.data.vision.domainnet.torchvision.datasets.ImageFolder",
            n_samples=10,
            domain="clipart",
            mode="train",
        )


@pytest.fixture
def mock_imagenet_dataset(sample_image: Image.Image) -> Dataset:
    with patch("waldo.data.vision.imagenet.Image.open", return_value=sample_image):
        return mock_dataset_image_folder_factory(
            ImageNetDataset,
            ImageNetDatasetConfig,
            patch_target="waldo.data.vision.imagenet.torchvision.datasets.ImageFolder",
            mode="train",
        )
