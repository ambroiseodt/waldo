r"""
Linear probing factory unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import pytest

from waldo.eval import LinearProbing, build_linear_probing


# -------------------------------------------------------------------
# Build linear probing
# -------------------------------------------------------------------
class TestBuildLinearProbing:
    r"""Tests for build_linear_probing factory function."""

    def test_config_is_correctly_passed(self) -> None:
        r"""build_linear_probing should return a LinearProbing instance with the given config."""
        probing = build_linear_probing({"tol": 1e-3, "max_iter": 10, "device": "cpu", "seed": 0})
        assert isinstance(probing, LinearProbing)
        assert (probing.tol, probing.max_iter, probing.device, probing.seed) == (1e-3, 10, "cpu", 0)

    def test_invalid_config_raises(self) -> None:
        r"""build_linear_probing should raise an error for an invalid config."""
        with pytest.raises(ValueError):
            build_linear_probing({"tol": -1.0})
