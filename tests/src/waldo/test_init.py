r"""
Package exports unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import importlib

import pytest

PACKAGES = ["waldo.data", "waldo.eval", "waldo.infra", "waldo.models"]


# -------------------------------------------------------------------
# Exports
# -------------------------------------------------------------------
@pytest.mark.parametrize("package", PACKAGES)
def test_all_exports_are_importable(package: str) -> None:
    r"""Every name listed in __all__, including the lazily imported ones, should be importable."""
    module = importlib.import_module(package)
    for name in module.__all__:
        assert getattr(module, name) is not None, name


@pytest.mark.parametrize("package", ["waldo.data", "waldo.models"])
def test_lazy_exports(package: str) -> None:
    r"""Lazily imported packages should list their exports and raise an AttributeError for unknown names."""
    module = importlib.import_module(package)
    assert dir(module) == sorted(module.__all__)
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = module.unexisting
