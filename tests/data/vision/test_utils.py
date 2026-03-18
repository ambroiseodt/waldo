r"""
Utils unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from waldo.data.vision.utils import build_dataset, build_loader, build_transform, deterministic_split


# -------------------------------------------------------------------
# deterministic_split
# -------------------------------------------------------------------
class TestDeterministicSplit:
    r"""Tests for deterministic_split."""

    def test_train_split_size(self) -> None:
        r"""Train split should contain 80% of the data."""
        data = np.arange(100)
        indices = deterministic_split(data, train_size=0.8, train=True)
        assert len(indices) == 80

    def test_test_split_size(self) -> None:
        r"""Test split should contain 20% of the data."""
        data = np.arange(100)
        indices = deterministic_split(data, train_size=0.8, train=False)
        assert len(indices) == 20

    def test_train_and_test_are_disjoint(self) -> None:
        r"""Train and test splits should not share any indices."""
        data = np.arange(100)
        train_indices = deterministic_split(data, train_size=0.8, train=True)
        test_indices = deterministic_split(data, train_size=0.8, train=False)
        assert len(np.intersect1d(train_indices, test_indices)) == 0

    def test_train_and_test_cover_all_indices(self) -> None:
        r"""Train and test splits together should cover all indices."""
        data = np.arange(100)
        train_indices = deterministic_split(data, train_size=0.8, train=True)
        test_indices = deterministic_split(data, train_size=0.8, train=False)
        all_indices = np.sort(np.concatenate([train_indices, test_indices]))
        assert np.array_equal(all_indices, np.arange(100))

    def test_split_is_deterministic(self) -> None:
        r"""Same data should always produce the same split."""
        data = np.arange(100)
        indices_a = deterministic_split(data, train_size=0.8, train=True)
        indices_b = deterministic_split(data, train_size=0.8, train=True)
        assert np.array_equal(indices_a, indices_b)

    def test_split_does_not_affect_global_random_state(self) -> None:
        r"""deterministic_split should restore the global numpy random state after execution."""
        data = np.arange(100)
        np.random.seed(0)
        state_before = np.random.get_state()[1].copy()
        deterministic_split(data, train_size=0.8, train=True)
        state_after = np.random.get_state()[1]
        assert np.array_equal(state_before, state_after)

    def test_indices_are_sorted(self) -> None:
        r"""Returned indices should be sorted in ascending order."""
        data = np.arange(100)
        indices = deterministic_split(data, train_size=0.8, train=True)
        assert np.array_equal(indices, np.sort(indices))


# -------------------------------------------------------------------
# build_transform
# -------------------------------------------------------------------
class TestBuildTransform:
    r"""Tests for build_transform."""

    @pytest.mark.parametrize("transform_type", ["train", "test", "static"])
    def test_returns_compose(self, transform_type: str) -> None:
        r"""build_transform should return a Compose object for all valid types."""
        transform = build_transform(resolution=224, transform_type=transform_type)
        assert isinstance(transform, Compose)

    @pytest.mark.parametrize("transform_type", ["train", "test", "static"])
    def test_case_insensitive(self, transform_type: str) -> None:
        r"""build_transform should be case insensitive."""
        transform = build_transform(resolution=224, transform_type=transform_type.upper())
        assert isinstance(transform, Compose)

    def test_invalid_transform_type_raises_value_error(self) -> None:
        r"""Invalid transform type should raise a ValueError."""
        with pytest.raises(ValueError, match="not found"):
            build_transform(resolution=224, transform_type="unexisting")

    def test_train_transform_has_random_resized_crop(self) -> None:
        r"""Train transform should include RandomResizedCrop."""
        from torchvision.transforms import RandomResizedCrop

        transform = build_transform(resolution=224, transform_type="train")
        assert any(isinstance(t, RandomResizedCrop) for t in transform.transforms)

    def test_train_transform_has_random_horizontal_flip(self) -> None:
        r"""Train transform should include RandomHorizontalFlip."""
        from torchvision.transforms import RandomHorizontalFlip

        transform = build_transform(resolution=224, transform_type="train")
        assert any(isinstance(t, RandomHorizontalFlip) for t in transform.transforms)

    def test_test_transform_has_center_crop(self) -> None:
        r"""Test transform should include CenterCrop."""
        from torchvision.transforms import CenterCrop

        transform = build_transform(resolution=224, transform_type="test")
        assert any(isinstance(t, CenterCrop) for t in transform.transforms)

    def test_test_transform_has_resize(self) -> None:
        r"""Test transform should include Resize."""
        from torchvision.transforms import Resize

        transform = build_transform(resolution=224, transform_type="test")
        assert any(isinstance(t, Resize) for t in transform.transforms)

    @pytest.mark.parametrize("transform_type", ["test", "static"])
    def test_test_and_static_are_identical(self, transform_type: str) -> None:
        r"""Test and static transforms should have the same number and types of transforms."""
        test_transform = build_transform(resolution=224, transform_type="test")
        static_transform = build_transform(resolution=224, transform_type="static")
        test_types = [type(t) for t in test_transform.transforms]
        static_types = [type(t) for t in static_transform.transforms]
        assert test_types == static_types


# -------------------------------------------------------------------
# build_dataset
# -------------------------------------------------------------------
class TestBuildDataset:
    r"""Tests for build_dataset."""

    def _make_mock_dataset(self, n_classes: int = 10) -> MagicMock:
        r"""Create a mock dataset instance with n_classes."""
        mock = MagicMock(spec=Dataset)
        mock.n_classes = n_classes
        return mock

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "aircraft",
            "cifar10",
            "cifar100",
            "flowers102",
            "food101",
            "imagenet",
            "pet",
        ],
    )
    def test_simple_dataset_names(self, dataset_name: str) -> None:
        r"""build_dataset should return a dataset and n_classes for all simple dataset names."""
        mock_dataset = self._make_mock_dataset()
        with (
            patch("waldo.data.vision.utils.build_with_type_check") as mock_config,
            patch(f"waldo.data.vision.utils.{dataset_name.capitalize()}Dataset", create=True) as mock_cls,
        ):
            mock_cls.return_value = mock_dataset
            mock_config.return_value = MagicMock()
            dataset, n_classes = build_dataset({"dataset_name": dataset_name, "mode": "train"})
        assert n_classes == mock_dataset.n_classes

    @pytest.mark.parametrize(
        "dataset_name, expected_corruption, expected_severity",
        [
            ("cifar10_c-corruption-brightness-severity-1", "brightness", "1"),
            ("cifar100_c-corruption-fog-severity-3", "fog", "3"),
            ("imagenet_c-corruption-snow-severity-5", "snow", "5"),
        ],
    )
    def test_corruption_dataset_name_parsing(
        self, dataset_name: str, expected_corruption: str, expected_severity: str
    ) -> None:
        r"""Corruption dataset names should be parsed into corruption_type and corruption_severity."""
        captured_config = {}

        def capture_config(config_class, config):
            captured_config.update(config)
            return MagicMock()

        mock_dataset = self._make_mock_dataset()
        base_name = dataset_name.split("-corruption")[0]

        with (
            patch("waldo.data.vision.utils.build_with_type_check", side_effect=capture_config),
            patch(
                f"waldo.data.vision.{base_name}.{base_name.replace('_', '').capitalize()}Dataset", create=True
            ) as mock_cls,
        ):
            mock_cls.return_value = mock_dataset
            build_dataset({"dataset_name": dataset_name, "mode": "train"})

        assert captured_config["corruption_type"] == expected_corruption
        assert captured_config["corruption_severity"] == expected_severity

    @pytest.mark.parametrize(
        "dataset_name, expected_domain",
        [
            ("domainnet-clipart", "clipart"),
            ("domainnet-real", "real"),
        ],
    )
    def test_domainnet_name_parsing(self, dataset_name: str, expected_domain: str) -> None:
        r"""DomainNet dataset names should be parsed into domain."""
        captured_config = {}

        def capture_config(config_class, config):
            captured_config.update(config)
            return MagicMock()

        mock_dataset = self._make_mock_dataset()
        with (
            patch("waldo.data.vision.utils.build_with_type_check", side_effect=capture_config),
            patch("waldo.data.vision.domainnet.DomainNetDataset", create=True) as mock_cls,
        ):
            mock_cls.return_value = mock_dataset
            build_dataset({"dataset_name": dataset_name, "mode": "train"})

        assert captured_config["domain"] == expected_domain

    def test_invalid_dataset_name_raises_value_error(self) -> None:
        r"""Invalid dataset name should raise a ValueError."""
        with pytest.raises(ValueError, match="not found"):
            build_dataset({"dataset_name": "unexisting", "mode": "train"})


# -------------------------------------------------------------------
# build_loader
# -------------------------------------------------------------------
class TestBuildLoader:
    r"""Tests for build_loader."""

    def _make_mock_dataset(self, n_classes: int = 10) -> MagicMock:
        r"""Create a mock dataset instance with n_classes."""
        mock = MagicMock(spec=Dataset)
        mock.n_classes = n_classes
        mock.__len__ = MagicMock(return_value=100)
        return mock

    def test_returns_dataloader(self) -> None:
        r"""build_loader should return a DataLoader."""
        with patch("waldo.data.vision.utils.build_dataset", return_value=(self._make_mock_dataset(), 10)):
            loader = build_loader({"dataset_name": "cifar10", "mode": "train"})
        assert isinstance(loader, DataLoader)

    def test_returns_dataloader_and_n_classes_when_requested(self) -> None:
        r"""build_loader should return a tuple of DataLoader and n_classes when return_n_classes=True."""
        with patch("waldo.data.vision.utils.build_dataset", return_value=(self._make_mock_dataset(), 10)):
            loader, n_classes = build_loader({"dataset_name": "cifar10", "mode": "train"}, return_n_classes=True)
        assert isinstance(loader, DataLoader)
        assert n_classes == 10

    def test_train_mode_shuffles(self) -> None:
        r"""DataLoader should shuffle data in train mode."""
        with patch("waldo.data.vision.utils.build_dataset", return_value=(self._make_mock_dataset(), 10)):
            loader = build_loader({"dataset_name": "cifar10", "mode": "train"})
        assert loader.sampler.__class__.__name__ == "RandomSampler"

    def test_test_mode_does_not_shuffle(self) -> None:
        r"""DataLoader should not shuffle data in test mode."""
        with patch("waldo.data.vision.utils.build_dataset", return_value=(self._make_mock_dataset(), 10)):
            loader = build_loader({"dataset_name": "cifar10", "mode": "test"})
        assert loader.sampler.__class__.__name__ == "SequentialSampler"

    def test_default_batch_size(self) -> None:
        r"""Default batch size should be 128."""
        with patch("waldo.data.vision.utils.build_dataset", return_value=(self._make_mock_dataset(), 10)):
            loader = build_loader({"dataset_name": "cifar10", "mode": "train"})
        assert loader.batch_size == 128

    def test_custom_batch_size(self) -> None:
        r"""Custom batch size should be respected."""
        with patch("waldo.data.vision.utils.build_dataset", return_value=(self._make_mock_dataset(), 10)):
            loader = build_loader({"dataset_name": "cifar10", "mode": "train", "batch_size": 32})
        assert loader.batch_size == 32
