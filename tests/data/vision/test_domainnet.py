r"""
DomainNet unit tests.

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

from waldo.data import DomainNetDataset, DomainNetDatasetConfig, build_transform

# Default Parameters, see src/waldo/data/vision/domainnet.py
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestDomainNetDatasetConfig:
    r"""Tests for config class valid entries and default values."""

    def test_default_mode_is_train(self) -> None:
        r"""Default mode should be 'train'."""
        config = DomainNetDatasetConfig()
        assert config.mode == "train"

    def test_default_transform_is_none(self) -> None:
        r"""Default transform should be None."""
        config = DomainNetDatasetConfig()
        assert config.transform is None

    def test_default_save_dir_is_path(self) -> None:
        r"""save_dir should be a Path object by default."""
        config = DomainNetDatasetConfig()
        assert isinstance(config.save_dir, Path)

    def test_default_domain_is_clipart(self) -> None:
        r"""Default domain should be 'clipart'."""
        config = DomainNetDatasetConfig()
        assert config.domain == "clipart"

    @pytest.mark.parametrize("mode", ["train", "test"])
    def test_valid_modes(self, mode: str) -> None:
        r"""Both valid modes should instantiate without errors."""
        config = DomainNetDatasetConfig(mode=mode)
        assert config.mode == mode

    @pytest.mark.parametrize("domain", ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"])
    def test_valid_domains(self, domain: str) -> None:
        r"""All valid domains should instantiate without errors."""
        config = DomainNetDatasetConfig(domain=domain)
        assert config.domain == domain

    def test_invalid_mode_raises_value_error(self) -> None:
        r"""Invalid mode should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            DomainNetDatasetConfig(mode="unexisting")

    def test_invalid_domain_raises_value_error(self) -> None:
        r"""Invalid domain should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid corruption type"):
            DomainNetDatasetConfig(domain="unexisting")

    def test_unknown_kwargs_are_ignored(self) -> None:
        r"""Unknown kwargs should be silently ignored and not set as attributes."""
        config = DomainNetDatasetConfig(mode="train", unknown_arg=42)
        assert not hasattr(config, "unknown_arg")

    def test_custom_save_dir(self, tmp_path: Path) -> None:
        r"""Custom save_dir should be stored as provided."""
        config = DomainNetDatasetConfig(save_dir=tmp_path)
        assert config.save_dir == tmp_path


# -------------------------------------------------------------------
# Dataset class
# -------------------------------------------------------------------
class TestDomainNetDataset:
    r"""Tests for dataset structure and item retrieval."""

    @pytest.mark.parametrize("mode, ratio", [("train", TRAIN_RATIO), ("test", TEST_RATIO)])
    def test_length(
        self, mock_dataset_image_folder_factory: Callable[..., DomainNetDataset], mode: str, ratio: float
    ) -> None:
        r"""Dataset length should match the expected split size for the given mode."""
        dataset = mock_dataset_image_folder_factory(
            config_class=DomainNetDatasetConfig,
            dataset_class=DomainNetDataset,
            patch_target="waldo.data.vision.domainnet.torchvision.datasets.ImageFolder",
            domain="clipart",
            mode=mode,
        )
        assert len(dataset) == int(10 * ratio)

    def test_n_classes(self, mock_domainnet_dataset: DomainNetDataset) -> None:
        r"""Dataset should expose exactly 345 classes."""
        assert mock_domainnet_dataset.n_classes == 345

    def test_getitem_without_transform_returns_pil_image(
        self, mock_domainnet_dataset: DomainNetDataset, sample_image: Image.Image
    ) -> None:
        r"""Without a transform, __getitem__ should return a PIL image."""
        with patch("waldo.data.vision.domainnet.Image.open", return_value=sample_image):
            sample, _ = mock_domainnet_dataset[0]
        assert isinstance(sample, Image.Image)

    def test_getitem_without_transform_returns_int_label(
        self, mock_domainnet_dataset: DomainNetDataset, sample_image: Image.Image
    ) -> None:
        r"""Without a transform, __getitem__ should return a Python int label."""
        with patch("waldo.data.vision.domainnet.Image.open", return_value=sample_image):
            _, label = mock_domainnet_dataset[0]
        assert isinstance(label, int)

    def test_getitem_with_transform_returns_tensor(
        self, mock_domainnet_dataset: DomainNetDataset, sample_image: Image.Image
    ) -> None:
        r"""With a transform, __getitem__ should return a float tensor."""
        mock_domainnet_dataset.transform = build_transform(resolution=224, transform_type="test")
        with patch("waldo.data.vision.domainnet.Image.open", return_value=sample_image):
            sample, _ = mock_domainnet_dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.dtype == torch.float32

    def test_getitem_with_transform_returns_correct_shape(
        self, mock_domainnet_dataset: DomainNetDataset, sample_image: Image.Image
    ) -> None:
        r"""With a transform, the tensor should have shape (3, resolution, resolution)."""
        resolution = 224
        mock_domainnet_dataset.transform = build_transform(resolution=resolution, transform_type="test")
        with patch("waldo.data.vision.domainnet.Image.open", return_value=sample_image):
            sample, _ = mock_domainnet_dataset[0]
        assert sample.shape == (3, resolution, resolution)

    def test_all_labels_in_valid_range(
        self, mock_domainnet_dataset: DomainNetDataset, sample_image: Image.Image
    ) -> None:
        r"""All labels should be integers in [0, n_classes - 1]."""
        with patch("waldo.data.vision.domainnet.Image.open", return_value=sample_image):
            for _, label in mock_domainnet_dataset:
                assert 0 <= label < mock_domainnet_dataset.n_classes

    def test_split_is_deterministic(self, mock_dataset_image_folder_factory: Callable[..., DomainNetDataset]) -> None:
        r"""Two datasets with the same config should yield identical samples and labels."""
        kwargs = dict(
            config_class=DomainNetDatasetConfig,
            dataset_class=DomainNetDataset,
            patch_target="waldo.data.vision.domainnet.torchvision.datasets.ImageFolder",
            domain="clipart",
            mode="train",
        )
        dataset_a = mock_dataset_image_folder_factory(**kwargs)
        dataset_b = mock_dataset_image_folder_factory(**kwargs)
        assert np.array_equal(dataset_a.samples, dataset_b.samples)

    def test_repr_is_string(self, mock_domainnet_dataset: DomainNetDataset) -> None:
        r"""__repr__ should return a non-empty string."""
        result = repr(mock_domainnet_dataset)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_repr_contains_length(self, mock_domainnet_dataset: DomainNetDataset) -> None:
        r"""__repr__ should mention the number of images in the train split."""
        expected = str(int(10 * TRAIN_RATIO))
        assert expected in repr(mock_domainnet_dataset)
