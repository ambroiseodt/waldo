r"""
Utils unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, RandomHorizontalFlip, RandomResizedCrop, Resize

from waldo.data.vision.factory import _worker_init_fn, build_dataset, build_loader, build_transform, deterministic_split


# -------------------------------------------------------------------
#   Dataset
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
        _, state_before, _, _, _ = np.random.get_state()
        deterministic_split(data, train_size=0.8, train=True)
        _, state_after, _, _, _ = np.random.get_state()
        assert np.array_equal(state_before, state_after)

    def test_indices_are_sorted(self) -> None:
        r"""Returned indices should be sorted in ascending order."""
        data = np.arange(100)
        indices = deterministic_split(data, train_size=0.8, train=True)
        assert np.array_equal(indices, np.sort(indices))


class TestBuildDataset:
    r"""Tests for build_dataset."""

    def _make_mock_dataset(self, n_classes: int = 10) -> MagicMock:
        r"""Create a minimal dataset stub with n_classes."""
        mock = MagicMock(spec=Dataset)
        mock.n_classes = n_classes
        return mock

    @pytest.mark.parametrize(
        "dataset_name, module, cls_name",
        [
            ("aircraft", "waldo.data.vision.aircraft", "AircraftDataset"),
            ("cifar10", "waldo.data.vision.cifar10", "Cifar10Dataset"),
            ("cifar100", "waldo.data.vision.cifar100", "Cifar100Dataset"),
            ("dtd", "waldo.data.vision.dtd", "DTDDataset"),
            ("eurosat", "waldo.data.vision.eurosat", "EuroSATDataset"),
            ("flowers102", "waldo.data.vision.flowers102", "Flowers102Dataset"),
            ("food101", "waldo.data.vision.food101", "Food101Dataset"),
            ("imagenet", "waldo.data.vision.imagenet", "ImageNetDataset"),
            ("pet", "waldo.data.vision.pet", "PetDataset"),
        ],
    )
    def test_simple_dataset_returns_dataset_and_n_classes(self, dataset_name: str, module: str, cls_name: str) -> None:
        r"""build_dataset should return a dataset and n_classes for all simple dataset names."""
        mock_dataset = self._make_mock_dataset()
        with (
            patch("waldo.data.vision.utils.build_with_type_check", return_value=MagicMock()),
            patch(f"{module}.{cls_name}", return_value=mock_dataset),
        ):
            _, n_classes = build_dataset({"dataset_name": dataset_name, "mode": "train"})
        assert n_classes == mock_dataset.n_classes

    @pytest.mark.parametrize(
        "dataset_name, expected_corruption, expected_severity",
        [
            ("cifar10_c-corruption-brightness-severity-1", "brightness", "1"),
            ("cifar10_c-corruption-fog-severity-3", "fog", "3"),
            ("cifar100_c-corruption-snow-severity-5", "snow", "5"),
            ("imagenet_c-corruption-gaussian_noise-severity-2", "gaussian_noise", "2"),
        ],
    )
    def test_corruption_dataset_name_parsing(
        self, dataset_name: str, expected_corruption: str, expected_severity: str
    ) -> None:
        r"""Corruption dataset names should be parsed into corruption_type and corruption_severity."""
        captured = {}

        def capture(config_class: Callable[..., Any], config: dict[str, Any]):
            captured.update(config)
            return MagicMock()

        mock_dataset = self._make_mock_dataset()
        base = dataset_name.split("-corruption")[0]
        module_map = {
            "cifar10_c": ("waldo.data.vision.cifar10_c", "Cifar10CDataset"),
            "cifar100_c": ("waldo.data.vision.cifar100_c", "Cifar100CDataset"),
            "imagenet_c": ("waldo.data.vision.imagenet_c", "ImageNetCDataset"),
        }
        module, cls_name = module_map[base]
        with (
            patch("waldo.data.vision.utils.build_with_type_check", side_effect=capture),
            patch(f"{module}.{cls_name}", return_value=mock_dataset),
        ):
            build_dataset({"dataset_name": dataset_name, "mode": "train"})
        assert captured["corruption_type"] == expected_corruption
        assert captured["corruption_severity"] == expected_severity

    @pytest.mark.parametrize(
        "dataset_name, expected_domain",
        [
            ("domainnet-clipart", "clipart"),
            ("domainnet-real", "real"),
            ("domainnet-quickdraw", "quickdraw"),
        ],
    )
    def test_domainnet_name_parsing(self, dataset_name: str, expected_domain: str) -> None:
        r"""DomainNet dataset names should be parsed into domain."""
        captured = {}

        def capture(config_class: Callable[..., Any], config: dict[str, Any]):
            captured.update(config)
            return MagicMock()

        mock_dataset = self._make_mock_dataset()
        with (
            patch("waldo.data.vision.utils.build_with_type_check", side_effect=capture),
            patch("waldo.data.vision.domainnet.DomainNetDataset", return_value=mock_dataset),
        ):
            build_dataset({"dataset_name": dataset_name, "mode": "train"})
        assert captured["domain"] == expected_domain

    def test_invalid_dataset_name_raises_value_error(self) -> None:
        r"""Invalid dataset name should raise a ValueError."""
        with pytest.raises(ValueError, match="not found"):
            build_dataset({"dataset_name": "unexisting", "mode": "train"})


# -------------------------------------------------------------------
#   Data preprocessing
# -------------------------------------------------------------------
class TestBuildTransform:
    r"""Tests for build_transform."""

    @pytest.mark.parametrize("transform_type", ["train", "val", "test", "static"])
    def test_returns_compose(self, transform_type: str) -> None:
        r"""build_transform should return a Compose object for all valid types."""
        transform = build_transform(resolution=224, transform_type=transform_type)
        assert isinstance(transform, Compose)

    @pytest.mark.parametrize("transform_type", ["train", "val", "test", "static"])
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
        transform = build_transform(resolution=224, transform_type="train")
        assert any(isinstance(t, RandomResizedCrop) for t in transform.transforms)

    def test_train_transform_has_random_horizontal_flip(self) -> None:
        r"""Train transform should include RandomHorizontalFlip."""
        transform = build_transform(resolution=224, transform_type="train")
        assert any(isinstance(t, RandomHorizontalFlip) for t in transform.transforms)

    @pytest.mark.parametrize("transform_type", ["val", "test", "static"])
    def test_non_train_transforms_have_center_crop(self, transform_type: str) -> None:
        r"""Val, test and static transforms should include CenterCrop."""
        transform = build_transform(resolution=224, transform_type=transform_type)
        assert any(isinstance(t, CenterCrop) for t in transform.transforms)

    @pytest.mark.parametrize("transform_type", ["val", "test", "static"])
    def test_non_train_transforms_have_resize(self, transform_type: str) -> None:
        r"""Val, test and static transforms should include Resize."""
        transform = build_transform(resolution=224, transform_type=transform_type)
        assert any(isinstance(t, Resize) for t in transform.transforms)

    @pytest.mark.parametrize("transform_type", ["val", "test", "static"])
    def test_val_test_static_are_identical(self, transform_type: str) -> None:
        r"""Val, test and static transforms should have the same types of transforms."""
        val_types = [type(t) for t in build_transform(resolution=224, transform_type="val").transforms]
        other_types = [type(t) for t in build_transform(resolution=224, transform_type=transform_type).transforms]
        assert val_types == other_types


# -------------------------------------------------------------------
# Dataloader
# -------------------------------------------------------------------
def test_worker_init_fn_sets_num_threads() -> None:
    r"""_worker_init_fn should set the number of threads to 1."""
    _worker_init_fn(worker_id=0)
    assert torch.get_num_threads() == 1


class TestBuildLoader:
    r"""Tests for build_loader."""

    def _make_mock_dataset(self, n_classes: int = 10) -> MagicMock:
        r"""Create a minimal dataset stub with n_classes and __len__."""
        mock = MagicMock()
        mock.n_classes = n_classes
        mock.__len__ = MagicMock(return_value=100)
        return mock

    def test_returns_dataloader(self) -> None:
        r"""build_loader should return a DataLoader."""
        with patch("waldo.data.vision.utils.build_dataset", return_value=(self._make_mock_dataset(), 10)):
            loader = build_loader({"dataset_name": "cifar10", "mode": "train"}, n_workers=0)
        assert isinstance(loader, DataLoader)

    def test_returns_dataloader_and_n_classes_when_requested(self) -> None:
        r"""build_loader should return a tuple of DataLoader and n_classes when return_n_classes=True."""
        with patch("waldo.data.vision.utils.build_dataset", return_value=(self._make_mock_dataset(), 10)):
            loader, n_classes = build_loader(
                {"dataset_name": "cifar10", "mode": "train"}, n_workers=0, return_n_classes=True
            )
        assert isinstance(loader, DataLoader)
        assert n_classes == 10

    def test_train_mode_shuffles(self) -> None:
        r"""DataLoader should shuffle data in train mode."""
        with patch("waldo.data.vision.utils.build_dataset", return_value=(self._make_mock_dataset(), 10)):
            loader = cast(DataLoader, build_loader({"dataset_name": "cifar10", "mode": "train"}, n_workers=0))
        assert loader.sampler.__class__.__name__ == "RandomSampler"

    def test_test_mode_does_not_shuffle(self) -> None:
        r"""DataLoader should not shuffle data in test mode."""
        with patch("waldo.data.vision.utils.build_dataset", return_value=(self._make_mock_dataset(), 10)):
            loader = cast(DataLoader, build_loader({"dataset_name": "cifar10", "mode": "test"}, n_workers=0))
        assert loader.sampler.__class__.__name__ == "SequentialSampler"

    def test_default_batch_size(self) -> None:
        r"""Default batch size should be 128."""
        with patch("waldo.data.vision.utils.build_dataset", return_value=(self._make_mock_dataset(), 10)):
            loader = cast(DataLoader, build_loader({"dataset_name": "cifar10", "mode": "train"}, n_workers=0))
        assert loader.batch_size == 128

    def test_custom_batch_size(self) -> None:
        r"""Custom batch size should be respected."""
        with patch("waldo.data.vision.utils.build_dataset", return_value=(self._make_mock_dataset(), 10)):
            loader = cast(
                DataLoader, build_loader({"dataset_name": "cifar10", "mode": "train", "batch_size": 32}, n_workers=0)
            )
        assert loader.batch_size == 32

    def test_default_n_workers(self) -> None:
        r"""Default number of workers should be 8."""
        with patch("waldo.data.vision.utils.build_dataset", return_value=(self._make_mock_dataset(), 10)):
            loader = cast(DataLoader, build_loader({"dataset_name": "cifar10", "mode": "train"}))
        assert loader.num_workers == 8

    def test_custom_n_workers(self) -> None:
        r"""Custom number of workers should be respected."""
        with patch("waldo.data.vision.utils.build_dataset", return_value=(self._make_mock_dataset(), 10)):
            loader = cast(DataLoader, build_loader({"dataset_name": "cifar10", "mode": "train"}, n_workers=0))
        assert loader.num_workers == 0
