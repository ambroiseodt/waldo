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

from waldo.infra.config import set_seed


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
