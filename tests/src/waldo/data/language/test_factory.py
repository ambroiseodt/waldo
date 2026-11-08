r"""
MTEB factory unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import pytest

from waldo.data import MTEB, build_mteb


class TestBuildMTEB:
    r"""Tests for build_mteb factory function."""

    def test_returns_mteb_instance(self) -> None:
        r"""build_mteb should return an MTEB instance."""
        benchmark = build_mteb({"benchmark_name": "Europe", "task_type": "Classification"})
        assert isinstance(benchmark, MTEB)

    def test_config_is_correctly_passed(self) -> None:
        r"""build_mteb should correctly pass config values to MTEB."""
        benchmark = build_mteb({"benchmark_name": "Europe", "task_type": "Retrieval", "max_n_docs": 1024})
        assert benchmark.benchmark_name == "Europe"
        assert benchmark.task_type == "Retrieval"
        assert benchmark.max_n_docs == 1024

    def test_invalid_config_raises_error(self) -> None:
        r"""build_mteb should raise an error for invalid config."""
        with pytest.raises(ValueError):
            build_mteb({"benchmark_name": "unexisting"})
