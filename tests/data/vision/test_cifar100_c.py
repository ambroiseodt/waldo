r"""
Cifar100-C unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from waldo.data import Cifar100CDataset, Cifar100CDatasetConfig, build_transform

# Default Parameters, see src/waldo/data/vision/cifar100_c.py
N_SAMPLES_PER_SEVERITY = 10_000
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------


class TestCifar100CDatasetConfig:
    r"""Tests for config class valid entries and default values."""

    def test_default_mode_is_train(self) -> None:
        r"""Default mode should be 'train'."""
        config = Cifar100CDatasetConfig()
        assert config.mode == "train"

    def test_default_transform_is_none(self) -> None:
        r"""Default transform should be None."""
        config = Cifar100CDatasetConfig()
        assert config.transform is None

    def test_default_save_dir_is_path(self) -> None:
        r"""save_dir should be a Path object by default."""
        config = Cifar100CDatasetConfig()
        assert isinstance(config.save_dir, Path)

    def test_default_corruption_type(self) -> None:
        r"""Default corruption type should be 'brightness'."""
        config = Cifar100CDatasetConfig()
        assert config.corruption_type == "brightness"

    def test_default_corruption_severity(self) -> None:
        r"""Default corruption severity should be 1."""
        config = Cifar100CDatasetConfig()
        assert config.corruption_severity == 1

    @pytest.mark.parametrize("mode", ["train", "test"])
    def test_valid_modes(self, mode: str) -> None:
        r"""Both valid modes should instantiate without errors."""
        config = Cifar100CDatasetConfig(mode=mode)
        assert config.mode == mode

    @pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
    def test_valid_severities(self, severity: int) -> None:
        r"""All valid severities should instantiate without errors."""
        config = Cifar100CDatasetConfig(corruption_severity=severity)
        assert config.corruption_severity == severity

    def test_invalid_mode_raises_value_error(self) -> None:
        r"""Invalid mode should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="Invalid mode"):
            Cifar100CDatasetConfig(mode="unexisting")

    def test_invalid_severity_raises_value_error(self) -> None:
        r"""Invalid severity should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid severity"):
            Cifar100CDatasetConfig(corruption_severity=6)

    def test_invalid_corruption_type_raises_value_error(self) -> None:
        r"""Invalid corruption type should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid corruption type"):
            Cifar100CDatasetConfig(corruption_type="unexisting")

    def test_unknown_kwargs_are_ignored(self) -> None:
        r"""Unknown kwargs should be silently ignored and not set as attributes."""
        config = Cifar100CDatasetConfig(mode="train", unknown_arg=42)
        assert not hasattr(config, "unknown_arg")

    def test_custom_save_dir(self, tmp_path: Path) -> None:
        r"""Custom save_dir should be stored as provided."""
        config = Cifar100CDatasetConfig(save_dir=tmp_path)
        assert config.save_dir == tmp_path


# -------------------------------------------------------------------
# Dataset class
# -------------------------------------------------------------------
class TestCifar100CDataset:
    r"""Tests for dataset structure and item retrieval."""

    @pytest.mark.parametrize("mode, ratio", [("train", TRAIN_RATIO), ("test", TEST_RATIO)])
    def test_length(self, mock_dataset_numpy_factory: Callable[..., Cifar100CDataset], mode: str, ratio: float) -> None:
        r"""Dataset length should match the expected split size for the given mode."""
        dataset = mock_dataset_numpy_factory(
            config_class=Cifar100CDatasetConfig,
            dataset_class=Cifar100CDataset,
            n_samples=50_000,
            n_classes=100,
            corruption_type="brightness",
            corruption_severity=1,
            mode=mode,
        )
        assert len(dataset) == int(N_SAMPLES_PER_SEVERITY * ratio)

    @pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
    def test_severity_slicing(self, mock_dataset_numpy_factory: Callable[..., Cifar100CDataset], severity: int) -> None:
        r"""Each severity level should yield a train split of the correct size."""
        dataset = mock_dataset_numpy_factory(
            config_class=Cifar100CDatasetConfig,
            dataset_class=Cifar100CDataset,
            n_samples=50_000,
            n_classes=100,
            corruption_type="brightness",
            corruption_severity=severity,
            mode="train",
        )
        assert len(dataset) == int(N_SAMPLES_PER_SEVERITY * TRAIN_RATIO)

    def test_n_classes(self, mock_cifar100c_dataset: Cifar100CDataset) -> None:
        r"""Dataset should expose exactly 100 classes."""
        assert mock_cifar100c_dataset.n_classes == 100

    def test_getitem_without_transform_returns_pil_image(self, mock_cifar100c_dataset: Cifar100CDataset) -> None:
        r"""Without a transform, __getitem__ should return a PIL image."""
        sample, _ = mock_cifar100c_dataset[0]
        assert isinstance(sample, Image.Image)

    def test_getitem_without_transform_returns_int_label(self, mock_cifar100c_dataset: Cifar100CDataset) -> None:
        r"""
        Without a transform, __getitem__ should return an integer label.
        Since labels are loaded with np.array, the correct type is np.integer.
        """
        _, label = mock_cifar100c_dataset[0]
        assert isinstance(label, np.integer)
        assert int(label) == label

    def test_getitem_with_transform_returns_int_label(self, mock_cifar100c_dataset: Cifar100CDataset) -> None:
        r"""
        With a transform, __getitem__ should return an integer label.
        Since labels are loaded with np.array, the correct type is np.integer.
        """
        _, label = mock_cifar100c_dataset[0]
        assert isinstance(label, np.integer)
        assert int(label) == label

    def test_getitem_with_transform_returns_tensor(self, mock_cifar100c_dataset: Cifar100CDataset) -> None:
        r"""With a transform, __getitem__ should return a float tensor."""
        mock_cifar100c_dataset.transform = build_transform(resolution=224, transform_type="test")
        sample, _ = mock_cifar100c_dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.dtype == torch.float32

    def test_getitem_with_transform_returns_correct_shape(self, mock_cifar100c_dataset: Cifar100CDataset) -> None:
        r"""With a transform, the tensor should have shape (3, resolution, resolution)."""
        resolution = 32
        mock_cifar100c_dataset.transform = build_transform(resolution=resolution, transform_type="test")
        sample, _ = mock_cifar100c_dataset[0]
        assert sample.shape == (3, resolution, resolution)

    def test_all_labels_in_valid_range(self, mock_cifar100c_dataset: Cifar100CDataset) -> None:
        r"""All labels should be in [0, n_classes - 1]."""
        for _, label in mock_cifar100c_dataset:
            assert 0 <= int(label) < mock_cifar100c_dataset.n_classes

    def test_split_is_deterministic(self, mock_dataset_numpy_factory: Callable[..., Cifar100CDataset]) -> None:
        r"""Two datasets with the same config should yield identical indices."""
        kwargs = dict(
            config_class=Cifar100CDatasetConfig,
            dataset_class=Cifar100CDataset,
            n_classes=100,
            corruption_type="brightness",
            corruption_severity=1,
            mode="train",
        )
        dataset_a = mock_dataset_numpy_factory(**kwargs)
        dataset_b = mock_dataset_numpy_factory(**kwargs)
        assert np.array_equal(dataset_a.data, dataset_b.data)
        assert list(dataset_a.targets) == list(dataset_b.targets)

    def test_repr_is_string(self, mock_cifar100c_dataset: Cifar100CDataset) -> None:
        r"""__repr__ should return a non-empty string."""
        result = repr(mock_cifar100c_dataset)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_repr_contains_length(self, mock_cifar100c_dataset: Cifar100CDataset) -> None:
        r"""__repr__ should mention the number of images in the train split."""
        expected = str(int(10 * TRAIN_RATIO))
        assert expected in repr(mock_cifar100c_dataset)
