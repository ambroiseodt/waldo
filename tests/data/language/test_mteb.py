r"""
MTEB unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from unittest.mock import ANY, MagicMock, patch

import pytest

from waldo.data.language.mteb import MTEB, MTEBConfig, build_mteb


def make_mock_task(
    train_n_docs: int = 100, test_n_docs: int = 100, domains: list | None = None, languages: list | None = None
) -> MagicMock:
    r"""Create a minimal mock task with descriptive_stats, domains and languages."""
    task = MagicMock()
    task.metadata.descriptive_stats = {
        "train": {"num_samples": train_n_docs},
        "test": {"num_samples": test_n_docs},
    }
    task.metadata.domains = domains or ["News"]
    task.metadata.languages = languages or ["eng"]
    return task


# -------------------------------------------------------------------
#   Config
# -------------------------------------------------------------------
class TestMTEBConfig:
    r"""Tests for config class valid entries and default values."""

    def test_default_benchmark_name(self) -> None:
        r"""Default benchmark name should be 'Multilingual'."""
        config = MTEBConfig()
        assert config.benchmark_name == "Multilingual"

    def test_default_task_type(self) -> None:
        r"""Default task type should be 'Classification'."""
        config = MTEBConfig()
        assert config.task_type == "Classification"

    def test_default_max_n_docs(self) -> None:
        r"""Default max_n_docs should be 8192."""
        config = MTEBConfig()
        assert config.max_n_docs == 8192

    def test_max_n_docs_can_be_none(self) -> None:
        r"""max_n_docs should accept None."""
        config = MTEBConfig(max_n_docs=None)
        assert config.max_n_docs is None

    @pytest.mark.parametrize("benchmark_name", ["Multilingual", "Europe", "Indic", "eng, v2"])
    def test_valid_benchmark_names(self, benchmark_name: str) -> None:
        r"""All valid benchmark names should instantiate without errors."""
        config = MTEBConfig(benchmark_name=benchmark_name)
        assert config.benchmark_name == benchmark_name

    @pytest.mark.parametrize(
        "task_type",
        [
            "Retrieval",
            "Classification",
            "Clustering",
            "STS",
            "PairClassification",
            "Reranking",
            "Summarization",
            "InstructionRetrieval",
        ],
    )
    def test_valid_task_types(self, task_type: str) -> None:
        r"""All valid task types should instantiate without errors."""
        config = MTEBConfig(task_type=task_type)
        assert config.task_type == task_type

    def test_invalid_benchmark_name_raises_assertion_error(self) -> None:
        r"""Invalid benchmark name should raise an AssertionError."""
        with pytest.raises(AssertionError):
            MTEBConfig(benchmark_name="unexisting")

    def test_invalid_task_type_raises_assertion_error(self) -> None:
        r"""Invalid task type should raise an AssertionError."""
        with pytest.raises(AssertionError):
            MTEBConfig(task_type="unexisting")

    def test_unknown_kwargs_are_ignored(self) -> None:
        r"""Unknown kwargs should be silently ignored and not set as attributes."""
        config = MTEBConfig(unknown_arg=42)
        assert not hasattr(config, "unknown_arg")


# -------------------------------------------------------------------
# Benchmark
# -------------------------------------------------------------------
class TestMTEB:
    r"""Tests for MTEB benchmark structure and filtering logic."""

    def _make_benchmark(self, **config_kwargs) -> MTEB:
        r"""Create an MTEB instance with the given config kwargs."""
        config = MTEBConfig(**config_kwargs)
        return MTEB(config)

    def test_stores_benchmark_name(self) -> None:
        r"""MTEB should store the benchmark name from config."""
        benchmark = self._make_benchmark(benchmark_name="Europe")
        assert benchmark.benchmark_name == "Europe"

    def test_stores_task_type(self) -> None:
        r"""MTEB should store the task type from config."""
        benchmark = self._make_benchmark(task_type="Retrieval")
        assert benchmark.task_type == "Retrieval"

    def test_stores_max_n_docs(self) -> None:
        r"""MTEB should store max_n_docs from config."""
        benchmark = self._make_benchmark(max_n_docs=1024)
        assert benchmark.max_n_docs == 1024

    def test_tasks_filters_by_task_type(self) -> None:
        r"""tasks property should call filter_tasks with the correct task type."""
        mock_tasks = [make_mock_task(train_n_docs=100, test_n_docs=100)]
        with (
            patch("waldo.data.language.mteb.mteb.get_benchmark", return_value=MagicMock()),
            patch("waldo.data.language.mteb.mteb.filter_tasks", return_value=mock_tasks) as mock_filter,
        ):
            benchmark = self._make_benchmark(task_type="Retrieval", max_n_docs=None)
            _ = benchmark.tasks
        mock_filter.assert_called_once_with(ANY, task_types=["Retrieval"])

    def test_tasks_filters_by_max_n_docs(self) -> None:
        r"""tasks property should exclude tasks exceeding max_n_docs."""
        small_task = make_mock_task(train_n_docs=100, test_n_docs=100)
        large_task = make_mock_task(train_n_docs=10_000, test_n_docs=10_000)
        with (
            patch("waldo.data.language.mteb.mteb.get_benchmark", return_value=MagicMock()),
            patch("waldo.data.language.mteb.mteb.filter_tasks", return_value=[small_task, large_task]),
        ):
            benchmark = self._make_benchmark(max_n_docs=200)
            tasks = benchmark.tasks
        assert len(tasks) == 1
        assert tasks[0] is small_task

    def test_tasks_no_filter_when_max_n_docs_is_none(self) -> None:
        r"""tasks property should return all tasks when max_n_docs is None."""
        mock_tasks = [make_mock_task(train_n_docs=10_000), make_mock_task(train_n_docs=100)]
        with (
            patch("waldo.data.language.mteb.mteb.get_benchmark", return_value=MagicMock()),
            patch("waldo.data.language.mteb.mteb.filter_tasks", return_value=mock_tasks),
        ):
            benchmark = self._make_benchmark(max_n_docs=None)
            tasks = benchmark.tasks
        assert len(tasks) == 2

    def test_tasks_excludes_task_without_stats(self) -> None:
        r"""tasks property should exclude tasks with no descriptive_stats."""
        task_with_stats = make_mock_task(train_n_docs=100, test_n_docs=100)
        task_without_stats = MagicMock()
        task_without_stats.metadata.descriptive_stats = None
        with (
            patch("waldo.data.language.mteb.mteb.get_benchmark", return_value=MagicMock()),
            patch("waldo.data.language.mteb.mteb.filter_tasks", return_value=[task_with_stats, task_without_stats]),
        ):
            benchmark = self._make_benchmark(max_n_docs=200)
            tasks = benchmark.tasks
        assert len(tasks) == 1
        assert tasks[0] is task_with_stats

    def test_tasks_uses_test_n_docs_when_no_train(self) -> None:
        r"""tasks property should fall back to test_n_docs when train split is absent."""
        task = MagicMock()
        task.metadata.descriptive_stats = {"test": {"num_samples": 100}}
        task.metadata.domains = ["News"]
        task.metadata.languages = ["eng"]
        with (
            patch("waldo.data.language.mteb.mteb.get_benchmark", return_value=MagicMock()),
            patch("waldo.data.language.mteb.mteb.filter_tasks", return_value=[task]),
        ):
            benchmark = self._make_benchmark(max_n_docs=200)
            tasks = benchmark.tasks
        assert len(tasks) == 1

    def test_n_tasks(self) -> None:
        r"""n_tasks should return the number of tasks."""
        mock_tasks = [make_mock_task() for _ in range(3)]
        with (
            patch("waldo.data.language.mteb.mteb.get_benchmark", return_value=MagicMock()),
            patch("waldo.data.language.mteb.mteb.filter_tasks", return_value=mock_tasks),
        ):
            benchmark = self._make_benchmark(max_n_docs=None)
            assert benchmark.n_tasks == 3

    def test_domains(self) -> None:
        r"""domains should return the union of all task domains."""
        task_a = make_mock_task(domains=["News", "Web"])
        task_b = make_mock_task(domains=["Web", "Medical"])
        with (
            patch("waldo.data.language.mteb.mteb.get_benchmark", return_value=MagicMock()),
            patch("waldo.data.language.mteb.mteb.filter_tasks", return_value=[task_a, task_b]),
        ):
            benchmark = self._make_benchmark(max_n_docs=None)
            assert benchmark.domains == {"News", "Web", "Medical"}

    def test_languages(self) -> None:
        r"""languages should return the union of all task languages."""
        task_a = make_mock_task(languages=["eng", "fra"])
        task_b = make_mock_task(languages=["fra", "deu"])
        with (
            patch("waldo.data.language.mteb.mteb.get_benchmark", return_value=MagicMock()),
            patch("waldo.data.language.mteb.mteb.filter_tasks", return_value=[task_a, task_b]),
        ):
            benchmark = self._make_benchmark(max_n_docs=None)
            assert benchmark.languages == {"eng", "fra", "deu"}

    def test_domains_handles_none_domains(self) -> None:
        r"""domains should handle tasks with None domains gracefully."""
        task = make_mock_task()
        task.metadata.domains = None
        with (
            patch("waldo.data.language.mteb.mteb.get_benchmark", return_value=MagicMock()),
            patch("waldo.data.language.mteb.mteb.filter_tasks", return_value=[task]),
        ):
            benchmark = self._make_benchmark(max_n_docs=None)
            assert benchmark.domains == set()

    def test_languages_handles_none_languages(self) -> None:
        r"""languages should handle tasks with None languages gracefully."""
        task = make_mock_task()
        task.metadata.languages = None
        with (
            patch("waldo.data.language.mteb.mteb.get_benchmark", return_value=MagicMock()),
            patch("waldo.data.language.mteb.mteb.filter_tasks", return_value=[task]),
        ):
            benchmark = self._make_benchmark(max_n_docs=None)
            assert benchmark.languages == set()


# -------------------------------------------------------------------
# build_mteb
# -------------------------------------------------------------------
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
        with pytest.raises(AssertionError):
            build_mteb({"benchmark_name": "unexisting"})
