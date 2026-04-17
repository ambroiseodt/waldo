r"""
ImageNet-C unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from waldo.data import ImageNetCDataset, ImageNetCDatasetConfig, build_transform

from .conftest import TEST_RATIO, TRAIN_RATIO


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestImageNetCDatasetConfig:
    r"""Tests for config class valid entries and default values."""

    def test_default_mode_is_train(self) -> None:
        r"""Default mode should be 'train'."""
        config = ImageNetCDatasetConfig()
        assert config.mode == "train"

    def test_default_transform_is_none(self) -> None:
        r"""Default transform should be None."""
        config = ImageNetCDatasetConfig()
        assert config.transform is None

    def test_default_save_dir_is_path(self) -> None:
        r"""save_dir should be a Path object by default."""
        config = ImageNetCDatasetConfig()
        assert isinstance(config.save_dir, Path)

    def test_default_corruption_type(self) -> None:
        r"""Default corruption type should be 'brightness'."""
        config = ImageNetCDatasetConfig()
        assert config.corruption_type == "brightness"

    def test_default_corruption_severity(self) -> None:
        r"""Default corruption severity should be 1."""
        config = ImageNetCDatasetConfig()
        assert config.corruption_severity == 1

    @pytest.mark.parametrize("mode", ["train", "test"])
    def test_valid_modes(self, mode: str) -> None:
        r"""Both valid modes should instantiate without errors."""
        config = ImageNetCDatasetConfig(mode=mode)
        assert config.mode == mode

    @pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
    def test_valid_severities(self, severity: int) -> None:
        r"""All valid severities should instantiate without errors."""
        config = ImageNetCDatasetConfig(corruption_severity=severity)
        assert config.corruption_severity == severity

    def test_invalid_mode_raises_value_error(self) -> None:
        r"""Invalid mode should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            ImageNetCDatasetConfig(mode="unexisting")

    def test_invalid_severity_raises_value_error(self) -> None:
        r"""Invalid severity should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid severity"):
            ImageNetCDatasetConfig(corruption_severity=6)

    def test_invalid_corruption_type_raises_value_error(self) -> None:
        r"""Invalid corruption type should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid corruption type"):
            ImageNetCDatasetConfig(corruption_type="unexisting")

    def test_unknown_kwargs_are_ignored(self) -> None:
        r"""Unknown kwargs should be silently ignored and not set as attributes."""
        config = ImageNetCDatasetConfig(mode="train", unknown_arg=42)
        assert not hasattr(config, "unknown_arg")

    def test_custom_save_dir(self, tmp_path: Path) -> None:
        r"""Custom save_dir should be stored as provided."""
        config = ImageNetCDatasetConfig(save_dir=tmp_path)
        assert config.save_dir == tmp_path


# -------------------------------------------------------------------
# Dataset class
# -------------------------------------------------------------------
class TestImageNetCDataset:
    r"""Tests for dataset structure and item retrieval."""

    @pytest.mark.parametrize("mode, ratio", [("train", TRAIN_RATIO), ("test", TEST_RATIO)])
    def test_length(
        self,
        mock_dataset_image_folder_factory: Callable[..., ImageNetCDataset],
        mode: str,
        ratio: float,
    ) -> None:
        r"""Dataset length should match the expected split size for the given mode."""
        dataset = mock_dataset_image_folder_factory(
            config_class=ImageNetCDatasetConfig,
            dataset_class=ImageNetCDataset,
            patch_target="waldo.data.vision.imagenet_c.torchvision.datasets.ImageFolder",
            n_samples=10,
            corruption_type="brightness",
            corruption_severity=1,
            mode=mode,
        )
        assert len(dataset) == int(10 * ratio)

    def test_n_classes(self, mock_imagenetc_dataset: ImageNetCDataset) -> None:
        r"""Dataset should expose exactly 1000 classes."""
        assert mock_imagenetc_dataset.n_classes == 1000

    def test_getitem_without_transform_returns_pil_image(
        self, mock_imagenetc_dataset: ImageNetCDataset, sample_image: Image.Image
    ) -> None:
        r"""Without a transform, __getitem__ should return a PIL image."""
        with patch("waldo.data.vision.imagenet_c.Image.open", return_value=sample_image):
            sample, _ = mock_imagenetc_dataset[0]
        assert isinstance(sample, Image.Image)

    def test_getitem_without_transform_returns_int_label(
        self, mock_imagenetc_dataset: ImageNetCDataset, sample_image: Image.Image
    ) -> None:
        r"""Without a transform, __getitem__ should return a Python int label."""
        with patch("waldo.data.vision.imagenet_c.Image.open", return_value=sample_image):
            _, label = mock_imagenetc_dataset[0]
        assert isinstance(label, int)

    def test_getitem_with_transform_returns_tensor(
        self, mock_imagenetc_dataset: ImageNetCDataset, sample_image: Image.Image
    ) -> None:
        r"""With a transform, __getitem__ should return a float tensor."""
        mock_imagenetc_dataset.transform = build_transform(resolution=224, transform_type="test")
        with patch("waldo.data.vision.imagenet_c.Image.open", return_value=sample_image):
            sample, _ = mock_imagenetc_dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.dtype == torch.float32

    def test_getitem_with_transform_returns_correct_shape(
        self, mock_imagenetc_dataset: ImageNetCDataset, sample_image: Image.Image
    ) -> None:
        r"""With a transform, the tensor should have shape (3, resolution, resolution)."""
        resolution = 224
        mock_imagenetc_dataset.transform = build_transform(resolution=resolution, transform_type="test")
        with patch("waldo.data.vision.imagenet_c.Image.open", return_value=sample_image):
            sample, _ = mock_imagenetc_dataset[0]
        assert sample.shape == (3, resolution, resolution)

    def test_all_labels_in_valid_range(
        self, mock_imagenetc_dataset: ImageNetCDataset, sample_image: Image.Image
    ) -> None:
        r"""All labels should be integers in [0, n_classes - 1]."""
        with patch("waldo.data.vision.imagenet_c.Image.open", return_value=sample_image):
            for _, label in mock_imagenetc_dataset:
                assert 0 <= label < mock_imagenetc_dataset.n_classes

    def test_split_is_deterministic(self, mock_dataset_image_folder_factory: Callable[..., ImageNetCDataset]) -> None:
        r"""Two datasets with the same config should yield identical samples."""
        kwargs = dict(
            config_class=ImageNetCDatasetConfig,
            dataset_class=ImageNetCDataset,
            patch_target="waldo.data.vision.imagenet_c.torchvision.datasets.ImageFolder",
            corruption_type="brightness",
            corruption_severity=1,
            mode="train",
        )
        dataset_a = mock_dataset_image_folder_factory(**kwargs)
        dataset_b = mock_dataset_image_folder_factory(**kwargs)
        assert np.array_equal(dataset_a.samples, dataset_b.samples)

    def test_repr_is_string(self, mock_imagenetc_dataset: ImageNetCDataset) -> None:
        r"""__repr__ should return a non-empty string."""
        result = repr(mock_imagenetc_dataset)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_repr_contains_length(self, mock_imagenetc_dataset: ImageNetCDataset) -> None:
        r"""__repr__ should mention the number of images in the train split."""
        expected = str(int(10 * TRAIN_RATIO))
        assert expected in repr(mock_imagenetc_dataset)
