r"""
Config unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import numpy as np
import torch

from waldo.infra.config import DEVICE, ROOT_DIR, SAVING_DIR, set_seed


# -------------------------------------------------------------------
# Paths and device
# -------------------------------------------------------------------
class TestPaths:
    r"""Tests for the default paths and device."""

    def test_root_dir_is_repository_root(self) -> None:
        r"""ROOT_DIR should be the repository root, which contains pyproject.toml."""
        assert (ROOT_DIR / "pyproject.toml").is_file()

    def test_saving_dir_is_under_root(self) -> None:
        r"""Runs should be saved under the repository root."""
        assert SAVING_DIR == ROOT_DIR / "savings"

    def test_device_matches_cuda_availability(self) -> None:
        r"""DEVICE should be cuda if available and cpu otherwise."""
        assert DEVICE == ("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
class TestSetSeed:
    r"""Tests for set_seed function."""

    def test_returns_rng(self) -> None:
        r"""set_seed should return a numpy Generator."""
        rng = set_seed(42)
        assert isinstance(rng, np.random.Generator)

    def test_rng_is_reproducible(self) -> None:
        r"""Two RNGs with the same seed should produce the same values."""
        rng_a = set_seed(42)
        rng_b = set_seed(42)
        assert rng_a.random() == rng_b.random()

    def test_numpy_seed_is_set(self) -> None:
        r"""set_seed should set the numpy random seed reproducibly."""
        set_seed(42)
        value_a = np.random.rand()
        set_seed(42)
        value_b = np.random.rand()
        assert value_a == value_b

    def test_torch_seed_is_set(self) -> None:
        r"""set_seed should set the torch random seed reproducibly."""
        set_seed(42)
        value_a = torch.rand(1).item()
        set_seed(42)
        value_b = torch.rand(1).item()
        assert value_a == value_b
