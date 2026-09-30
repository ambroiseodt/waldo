r"""
RxRx1 unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import pytest
import torch
from PIL import Image

from waldo.data import RxRx1Dataset, RxRx1DatasetConfig, build_transform


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestRxRx1DatasetConfig:
    r"""Tests for config class valid entries and default values."""

    def test_default_mode_is_train(self) -> None:
        r"""Default mode should be 'train'."""
        config = RxRx1DatasetConfig()
        assert config.mode == "train"

    def test_default_transform_is_none(self) -> None:
        r"""Default transform should be None."""
        config = RxRx1DatasetConfig()
        assert config.transform is None

    @pytest.mark.parametrize("mode", ["train", "test"])
    def test_valid_modes(self, mode: str) -> None:
        r"""Both valid modes should instantiate without errors."""
        config = RxRx1DatasetConfig(mode=mode)
        assert config.mode == mode

    def test_invalid_mode_raises_value_error(self) -> None:
        r"""Invalid mode should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="Invalid mode"):
            RxRx1DatasetConfig(mode="unexisting")

    def test_unknown_kwargs_are_ignored(self) -> None:
        r"""Unknown kwargs should be silently ignored and not set as attributes."""
        config = RxRx1DatasetConfig(mode="train", unknown_arg=42)
        assert not hasattr(config, "unknown_arg")


# -------------------------------------------------------------------
# Dataset class
# -------------------------------------------------------------------
class TestRxRx1Dataset:
    r"""Tests for dataset structure and item retrieval."""

    def test_loads_wilds_subset(self, mock_rxrx1_dataset: RxRx1Dataset) -> None:
        r"""The dataset should be loaded from WILDS and restricted to the subset of the mode."""
        mock_get_dataset = mock_rxrx1_dataset.mock_get_dataset
        assert mock_get_dataset.call_args.kwargs["dataset"] == "rxrx1"
        mock_get_dataset.return_value.get_subset.assert_called_once_with("test")

    def test_length(self, mock_rxrx1_dataset: RxRx1Dataset) -> None:
        r"""Dataset length should match the number of samples."""
        assert len(mock_rxrx1_dataset) == 10

    def test_n_classes(self, mock_rxrx1_dataset: RxRx1Dataset) -> None:
        r"""Dataset should expose exactly 1139 classes."""
        assert mock_rxrx1_dataset.n_classes == 1139

    def test_getitem_without_transform_returns_pil_image(self, mock_rxrx1_dataset: RxRx1Dataset) -> None:
        r"""Without a transform, __getitem__ should return a PIL image."""
        sample, _ = mock_rxrx1_dataset[0]
        assert isinstance(sample, Image.Image)

    def test_getitem_without_transform_returns_int_label(self, mock_rxrx1_dataset: RxRx1Dataset) -> None:
        r"""Without a transform, __getitem__ should return an integer label and drop the metadata."""
        item = mock_rxrx1_dataset[3]
        assert len(item) == 2
        assert item[1] == 3

    def test_getitem_with_transform_returns_correct_shape(self, mock_rxrx1_dataset: RxRx1Dataset) -> None:
        r"""With a transform, the tensor should have shape (3, resolution, resolution)."""
        resolution = 224
        mock_rxrx1_dataset.transform = build_transform(resolution=resolution, transform_type="test")
        sample, _ = mock_rxrx1_dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (3, resolution, resolution)

    def test_repr_contains_length(self, mock_rxrx1_dataset: RxRx1Dataset) -> None:
        r"""__repr__ should mention the number of images. Here, the mock dataset contains 10 images."""
        assert "10" in repr(mock_rxrx1_dataset)
