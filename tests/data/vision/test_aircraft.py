r"""
FGVCAircraft unit tests.

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

from waldo.data import FGVCAircraftDataset, FGVCAircraftDatasetConfig, build_transform

# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------


class TestFGVCFGVCAircraftDatasetConfig:
    r"""Tests for config class valid entries and default values."""

    def test_default_mode_is_train(self) -> None:
        r"""Default mode should be 'train'."""
        config = FGVCAircraftDatasetConfig()
        assert config.mode == "train"

    def test_default_transform_is_none(self) -> None:
        r"""Default transform should be None."""
        config = FGVCAircraftDatasetConfig()
        assert config.transform is None

    def test_default_save_dir_is_path(self) -> None:
        r"""save_dir should be a Path object by default."""
        config = FGVCAircraftDatasetConfig()
        assert isinstance(config.save_dir, Path)

    @pytest.mark.parametrize("mode", ["train", "test"])
    def test_valid_modes(self, mode: str) -> None:
        r"""Both valid modes should instantiate without errors."""
        config = FGVCAircraftDatasetConfig(mode=mode)
        assert config.mode == mode

    def test_invalid_mode_raises_value_error(self) -> None:
        r"""Invalid mode should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="Invalid mode"):
            FGVCAircraftDatasetConfig(mode="unexisting")

    def test_unknown_kwargs_are_ignored(self) -> None:
        r"""Unknown kwargs should be silently ignored and not set as attributes."""
        config = FGVCAircraftDatasetConfig(mode="train", unknown_arg=42)
        assert not hasattr(config, "unknown_arg")

    def test_custom_save_dir(self, tmp_path: Path) -> None:
        r"""Custom save_dir should be stored as provided."""
        config = FGVCAircraftDatasetConfig(save_dir=tmp_path)
        assert config.save_dir == tmp_path


# -------------------------------------------------------------------
# Dataset class
# -------------------------------------------------------------------


class TestFGVCFGVCAircraftDataset:
    r"""Tests for dataset structure and item retrieval."""

    def test_length(self, mock_aircraft_dataset: FGVCAircraftDataset) -> None:
        r"""Dataset length should match the number of samples."""
        assert len(mock_aircraft_dataset) == 10

    def test_n_classes(self, mock_aircraft_dataset: FGVCAircraftDataset) -> None:
        r"""Dataset should expose exactly 100 classes."""
        assert mock_aircraft_dataset.n_classes == 100

    def test_getitem_without_transform_returns_pil_image(self, mock_aircraft_dataset: FGVCAircraftDataset) -> None:
        r"""Without a transform, __getitem__ should return a PIL image."""
        sample, _ = mock_aircraft_dataset[0]
        assert isinstance(sample, Image.Image)

    def test_getitem_without_transform_returns_int_label(self, mock_aircraft_dataset: FGVCAircraftDataset) -> None:
        r"""Without a transform, __getitem__ should return an integer label."""
        _, label = mock_aircraft_dataset[0]
        assert isinstance(label, int)

    def test_getitem_with_transform_returns_int_label(self, mock_aircraft_dataset: FGVCAircraftDataset) -> None:
        r"""With a transform, __getitem__ should return an integer label."""
        mock_aircraft_dataset.transform = build_transform(resolution=224, transform_type="test")
        _, label = mock_aircraft_dataset[0]
        assert isinstance(label, int)

    def test_getitem_with_transform_returns_tensor(self, mock_aircraft_dataset: FGVCAircraftDataset) -> None:
        r"""With a transform, __getitem__ should return a float tensor."""
        mock_aircraft_dataset.transform = build_transform(resolution=224, transform_type="test")
        sample, _ = mock_aircraft_dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.dtype == torch.float32

    def test_getitem_with_transform_returns_correct_shape(self, mock_aircraft_dataset: FGVCAircraftDataset) -> None:
        r"""With a transform, the tensor should have shape (3, resolution, resolution)."""
        resolution = 224
        mock_aircraft_dataset.transform = build_transform(resolution=resolution, transform_type="test")
        sample, _ = mock_aircraft_dataset[0]
        assert sample.shape == (3, resolution, resolution)

    def test_all_labels_in_valid_range(self, mock_aircraft_dataset: FGVCAircraftDataset) -> None:
        r"""All labels should be integers in [0, n_classes - 1]."""
        for _, label in mock_aircraft_dataset:
            assert 0 <= label < mock_aircraft_dataset.n_classes

    def test_repr_is_string(self, mock_aircraft_dataset: FGVCAircraftDataset) -> None:
        r"""__repr__ should return a non-empty string."""
        result = repr(mock_aircraft_dataset)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_repr_contains_length(self, mock_aircraft_dataset: FGVCAircraftDataset) -> None:
        r"""__repr__ should mention the number of images. Here, the mock dataset contains 10 images."""
        result = repr(mock_aircraft_dataset)
        assert "10" in result
