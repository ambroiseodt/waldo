r"""
Food101 unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from pathlib import Path

import pytest
import torch
from PIL import Image

from waldo.data import Food101Dataset, Food101DatasetConfig, build_transform


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestFood101DatasetConfig:
    r"""Tests for config class valid entries and default values."""

    def test_default_mode_is_train(self) -> None:
        r"""Default mode should be 'train'."""
        config = Food101DatasetConfig()
        assert config.mode == "train"

    def test_default_transform_is_none(self) -> None:
        r"""Default transform should be None."""
        config = Food101DatasetConfig()
        assert config.transform is None

    def test_default_save_dir_is_path(self) -> None:
        r"""save_dir should be a Path object by default."""
        config = Food101DatasetConfig()
        assert isinstance(config.save_dir, Path)

    @pytest.mark.parametrize("mode", ["train", "test"])
    def test_valid_modes(self, mode: str) -> None:
        r"""Both valid modes should instantiate without errors."""
        config = Food101DatasetConfig(mode=mode)
        assert config.mode == mode

    def test_invalid_mode_raises_value_error(self) -> None:
        r"""Invalid mode should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="Invalid mode"):
            Food101DatasetConfig(mode="unexisting")

    def test_unknown_kwargs_are_ignored(self) -> None:
        r"""Unknown kwargs should be silently ignored and not set as attributes."""
        config = Food101DatasetConfig(mode="train", unknown_arg=42)
        assert not hasattr(config, "unknown_arg")

    def test_custom_save_dir(self, tmp_path: Path) -> None:
        r"""Custom save_dir should be stored as provided."""
        config = Food101DatasetConfig(save_dir=tmp_path)
        assert config.save_dir == tmp_path


# -------------------------------------------------------------------
# Dataset class
# -------------------------------------------------------------------
class TestFood101Dataset:
    r"""Tests for dataset structure and item retrieval."""

    def test_length(self, mock_food101_dataset: Food101Dataset) -> None:
        r"""Dataset length should match the number of samples."""
        assert len(mock_food101_dataset) == 10

    def test_n_classes(self, mock_food101_dataset: Food101Dataset) -> None:
        r"""Dataset should expose exactly 101 classes."""
        assert mock_food101_dataset.n_classes == 101

    def test_getitem_without_transform_returns_pil_image(self, mock_food101_dataset: Food101Dataset) -> None:
        r"""Without a transform, __getitem__ should return a PIL image."""
        sample, _ = mock_food101_dataset[0]
        assert isinstance(sample, Image.Image)

    def test_getitem_without_transform_returns_int_label(self, mock_food101_dataset: Food101Dataset) -> None:
        r"""Without a transform, __getitem__ should return an integer label."""
        _, label = mock_food101_dataset[0]
        assert isinstance(label, int)

    def test_getitem_with_transform_returns_int_label(self, mock_food101_dataset: Food101Dataset) -> None:
        r"""With a transform, __getitem__ should return an integer label."""
        mock_food101_dataset.transform = build_transform(resolution=224, transform_type="test")
        _, label = mock_food101_dataset[0]
        assert isinstance(label, int)

    def test_getitem_with_transform_returns_tensor(self, mock_food101_dataset: Food101Dataset) -> None:
        r"""With a transform, __getitem__ should return a float tensor."""
        mock_food101_dataset.transform = build_transform(resolution=224, transform_type="test")
        sample, _ = mock_food101_dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.dtype == torch.float32

    def test_getitem_with_transform_returns_correct_shape(self, mock_food101_dataset: Food101Dataset) -> None:
        r"""With a transform, the tensor should have shape (3, resolution, resolution)."""
        resolution = 224
        mock_food101_dataset.transform = build_transform(resolution=resolution, transform_type="test")
        sample, _ = mock_food101_dataset[0]
        assert sample.shape == (3, resolution, resolution)

    def test_all_labels_in_valid_range(self, mock_food101_dataset: Food101Dataset) -> None:
        r"""All labels should be integers in [0, n_classes - 1]."""
        for _, label in mock_food101_dataset:
            assert 0 <= label < mock_food101_dataset.n_classes

    def test_repr_is_string(self, mock_food101_dataset: Food101Dataset) -> None:
        r"""__repr__ should return a non-empty string."""
        result = repr(mock_food101_dataset)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_repr_contains_length(self, mock_food101_dataset: Food101Dataset) -> None:
        r"""__repr__ should mention the number of images. Here, the mock dataset contains 10 images."""
        result = repr(mock_food101_dataset)
        assert "10" in result
