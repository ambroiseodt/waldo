r"""
Massive Text Embedding Benchmark (MTEB).

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
from dataclasses import dataclass
from functools import cached_property

import mteb
from mteb.models.models_protocols import EncoderProtocol
from mteb.results import ModelResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
#   Config
# ------------------------------------------------------------------------------
@dataclass
class MTEBConfig:
    r"""
    MTEB configuration file.

    Parameters
    ----------
    benchmark_name: str
        Name of the benchmark. Options are "Multilingual", "Europe", "Indic", "eng, v2".
        We focus on the 4 main benchmarks presented in [1]_, which are the "Multilingual", "Europe", "Indic"
        and "eng, v2".
    task_type: str
        Type of task. Options are
            "Retrieval",
            "Classification",
            "Clustering",
            "STS",
            "PairClassification",
            "Reranking",
            "Summarization",
            "InstructionRetrieval".
    max_n_docs: int | None
        Maximum number of documents to use. This allows us to filter out large tasks that are too
        costly to evaluate on. If None, all the tasks are used.
    batch_size: int, default=32
        Batch size for evaluation.

    References
    ----------
    .. [1] K. Enevoldsen et al. MMTEB: Massive Multilingual Text Embedding Benchmark. In ICLR 2025
    """

    benchmark_name: str = "Multilingual"
    task_type: str = "Classification"
    max_n_docs: int | None = 8192
    batch_size: int = 32

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Valid benchmark name
        benchmark_names = ["Multilingual", "Europe", "Indic", "eng, v2"]
        if self.benchmark_name not in benchmark_names:
            raise ValueError(f"Invalid benchmark name {self.benchmark_name}. Options are {benchmark_names}.")

        # Valid task type
        task_types = [
            "Retrieval",
            "Classification",
            "Clustering",
            "STS",
            "PairClassification",
            "Reranking",
            "Summarization",
            "InstructionRetrieval",
        ]
        if self.task_type not in task_types:
            raise ValueError(f"Invalid task type {self.task_type}. Options are {task_types}.")


# ------------------------------------------------------------------------------
#   Benchmark
# ------------------------------------------------------------------------------
class MTEB:
    r"""
    MTEB benchmark from [1, 2]_.

    It consists of over 500 quality-controlled evaluation tasks across more than 250 languages.
    We focus on the 4 main benchmarks presented in [1]_, which are the "Multilingual", "Europe", "Indic"
    and "eng, v2". For efficiency reasons,
    we also filter out large tasks that are too costly to evaluate on, by setting the max_n_docs parameter. It allows
    to select the task where the amount of training or test documents is below max_n_docs.

    Parameters
    ----------
    config: configuration class with
        benchmark_name: str, default="Multilingual"
            Name of the benchmark.
        task_type: str, default="Classification"
            Type of task.
        max_n_docs: int | None, default=8192
            Maximum number of documents to use.
        batch_size: int, default=32
            Batch size for evaluation.

    Notes
    -----
    Official website: https://github.com/embeddings-benchmark/mteb?tab=readme-ov-file.

    References
    ----------
    .. [1] N. Muennighoff et al. MTEB: Massive Text Embedding Benchmark. In EACL 2023

    .. [2] K. Enevoldsen et al. MMTEB: Massive Multilingual Text Embedding Benchmark. In ICLR 2025
    """

    def __init__(self, config: MTEBConfig):
        self.benchmark_name = config.benchmark_name
        self.task_type = config.task_type
        self.max_n_docs = config.max_n_docs
        self.batch_size = config.batch_size

    def evaluate(self, model: EncoderProtocol, cache_path: str) -> ModelResult:
        r"""
        Evaluate the model on the tasks of the benchmark. Results are saved to the cache path.

        Parameters
        ----------
        model: EncoderProtocol
            Model to evaluate, e.g., a BaseLanguageModel probed at a given layer and module.
        cache_path: str
            Path where the results are saved. Tasks already evaluated are loaded from the cache.

        Returns
        -------
        ModelResult
            Results of the evaluation.
        """
        cache = mteb.ResultCache(cache_path=cache_path)
        return mteb.evaluate(
            model,
            tasks=self.tasks,
            encode_kwargs={"batch_size": self.batch_size},
            cache=cache,
        )

    @cached_property
    def tasks(self) -> list:
        r"""
        Get the tasks of the benchmark.

        Returns
        -------
        tasks: list
            List of tasks of type task_type with at most max_n_docs documents.
        """
        # Recover benchmark
        benchmark = mteb.get_benchmark(f"MTEB({self.benchmark_name})")

        # Filter by task type
        tasks = mteb.filter_tasks(benchmark, task_types=[self.task_type])

        # Filter by number of documents if max_n_docs is set
        if self.max_n_docs is not None:
            reduced_tasks = []
            for task in tasks:
                if stats := task.metadata.descriptive_stats:
                    train_n_docs = stats["train"]["num_samples"] if "train" in stats else 0
                    test_n_docs = stats["test"]["num_samples"] if "test" in stats else 0
                    if max(train_n_docs, test_n_docs) <= self.max_n_docs:
                        reduced_tasks.append(task)
            tasks = reduced_tasks

        return tasks

    @property
    def n_tasks(self) -> int:
        r"""Return the number of tasks."""
        return len(self.tasks)

    @property
    def domains(self) -> set[str]:
        r"""Return the domains covered by the tasks."""
        return {domain for task in self.tasks for domain in (task.metadata.domains or [])}

    @property
    def languages(self) -> set[str]:
        r"""Return the languages covered by the tasks."""
        return {lang for task in self.tasks for lang in (task.metadata.languages or [])}
