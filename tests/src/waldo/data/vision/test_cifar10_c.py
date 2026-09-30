r"""
Cifar10-C unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from PIL import Image

from waldo.data import Cifar10CDataset, Cifar10CDatasetConfig, build_transform

from .conftest import N_SAMPLES_PER_SEVERITY, TEST_RATIO, TRAIN_RATIO


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestCifar10CDatasetConfig:
    r"""Tests for config class valid entries and default values."""

    def test_default_mode_is_train(self) -> None:
        r"""Default mode should be 'train'."""
        config = Cifar10CDatasetConfig()
        assert config.mode == "train"

    def test_default_transform_is_none(self) -> None:
        r"""Default transform should be None."""
        config = Cifar10CDatasetConfig()
        assert config.transform is None

    def test_default_corruption_type(self) -> None:
        r"""Default corruption type should be 'brightness'."""
        config = Cifar10CDatasetConfig()
        assert config.corruption_type == "brightness"

    def test_default_corruption_severity(self) -> None:
        r"""Default corruption severity should be 1."""
        config = Cifar10CDatasetConfig()
        assert config.corruption_severity == 1

    @pytest.mark.parametrize("mode", ["train", "test"])
    def test_valid_modes(self, mode: str) -> None:
        r"""Both valid modes should instantiate without errors."""
        config = Cifar10CDatasetConfig(mode=mode)
        assert config.mode == mode

    @pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
    def test_valid_severities(self, severity: int) -> None:
        r"""All valid severities should instantiate without errors."""
        config = Cifar10CDatasetConfig(corruption_severity=severity)
        assert config.corruption_severity == severity

    def test_invalid_mode_raises_value_error(self) -> None:
        r"""Invalid mode should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="Invalid mode"):
            Cifar10CDatasetConfig(mode="unexisting")

    def test_invalid_severity_raises_value_error(self) -> None:
        r"""Invalid severity should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid severity"):
            Cifar10CDatasetConfig(corruption_severity=6)

    def test_invalid_corruption_type_raises_value_error(self) -> None:
        r"""Invalid corruption type should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid corruption type"):
            Cifar10CDatasetConfig(corruption_type="unexisting")

    def test_unknown_kwargs_are_ignored(self) -> None:
        r"""Unknown kwargs should be silently ignored and not set as attributes."""
        config = Cifar10CDatasetConfig(mode="train", unknown_arg=42)
        assert not hasattr(config, "unknown_arg")


# -------------------------------------------------------------------
# Dataset class
# -------------------------------------------------------------------
class TestCifar10CDataset:
    r"""Tests for dataset structure and item retrieval."""

    @pytest.mark.parametrize("mode, ratio", [("train", TRAIN_RATIO), ("test", TEST_RATIO)])
    def test_length(self, mock_dataset_numpy_factory: Callable[..., Cifar10CDataset], mode: str, ratio: float) -> None:
        r"""Dataset length should match the expected split size for the given mode."""
        dataset = mock_dataset_numpy_factory(
            config_class=Cifar10CDatasetConfig,
            dataset_class=Cifar10CDataset,
            n_samples=50_000,
            n_classes=10,
            corruption_type="brightness",
            corruption_severity=1,
            mode=mode,
        )
        assert len(dataset) == int(N_SAMPLES_PER_SEVERITY * ratio)

    @pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
    def test_severity_slicing(self, mock_dataset_numpy_factory: Callable[..., Cifar10CDataset], severity: int) -> None:
        r"""Each severity level should yield a train split of the correct size."""
        dataset = mock_dataset_numpy_factory(
            config_class=Cifar10CDatasetConfig,
            dataset_class=Cifar10CDataset,
            n_samples=50_000,
            n_classes=10,
            corruption_type="brightness",
            corruption_severity=severity,
            mode="train",
        )
        assert len(dataset) == int(N_SAMPLES_PER_SEVERITY * TRAIN_RATIO)

    def test_n_classes(self, mock_cifar10c_dataset: Cifar10CDataset) -> None:
        r"""Dataset should expose exactly 10 classes."""
        assert mock_cifar10c_dataset.n_classes == 10

    def test_getitem_without_transform_returns_pil_image(self, mock_cifar10c_dataset: Cifar10CDataset) -> None:
        r"""Without a transform, __getitem__ should return a PIL image."""
        sample, _ = mock_cifar10c_dataset[0]
        assert isinstance(sample, Image.Image)

    def test_getitem_without_transform_returns_int_label(self, mock_cifar10c_dataset: Cifar10CDataset) -> None:
        r"""
        Without a transform, __getitem__ should return an integer label.
        Since labels are loaded with np.array, the correct type is np.integer.
        """
        _, label = mock_cifar10c_dataset[0]
        assert isinstance(label, np.integer)
        assert int(label) == label

    def test_getitem_with_transform_returns_tensor(self, mock_cifar10c_dataset: Cifar10CDataset) -> None:
        r"""With a transform, __getitem__ should return a float tensor."""
        mock_cifar10c_dataset.transform = build_transform(resolution=224, transform_type="test")
        sample, _ = mock_cifar10c_dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.dtype == torch.float32

    def test_getitem_with_transform_returns_correct_shape(self, mock_cifar10c_dataset: Cifar10CDataset) -> None:
        r"""With a transform, the tensor should have shape (3, resolution, resolution)."""
        resolution = 32
        mock_cifar10c_dataset.transform = build_transform(resolution=resolution, transform_type="test")
        sample, _ = mock_cifar10c_dataset[0]
        assert sample.shape == (3, resolution, resolution)

    def test_all_labels_in_valid_range(self, mock_cifar10c_dataset: Cifar10CDataset) -> None:
        r"""All labels should be in [0, n_classes - 1]."""
        for _, label in mock_cifar10c_dataset:
            assert 0 <= int(label) < mock_cifar10c_dataset.n_classes

    def test_split_is_deterministic(self, mock_dataset_numpy_factory: Callable[..., Cifar10CDataset]) -> None:
        r"""Two datasets with the same config should yield identical indices."""
        kwargs = dict(
            config_class=Cifar10CDatasetConfig,
            dataset_class=Cifar10CDataset,
            n_classes=10,
            corruption_type="brightness",
            corruption_severity=1,
            mode="train",
        )
        dataset_a = mock_dataset_numpy_factory(**kwargs)
        dataset_b = mock_dataset_numpy_factory(**kwargs)
        assert np.array_equal(dataset_a.data, dataset_b.data)
        assert list(dataset_a.targets) == list(dataset_b.targets)

    def test_repr_contains_length(self, mock_cifar10c_dataset: Cifar10CDataset) -> None:
        r"""__repr__ should mention the number of images in the train split."""
        expected = str(int(10 * TRAIN_RATIO))
        assert expected in repr(mock_cifar10c_dataset)
