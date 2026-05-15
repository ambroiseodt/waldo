r"""
Multilingual Text Embedding Benchmark.

Notes
-----
The sentence_transformer package is already a dependency of mteb and need not be installed separately.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property, wraps
from typing import Any

import mteb
from mteb.results import ModelResult
from sentence_transformers import SentenceTransformer

import datasets

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
        and "eng, v2". Other choices are possible, see the official website for more details.
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
        Maximul number of documents to use. This allows us to filter out large tasks that are too
        costly to evaluate on. If None, all the tasks are used.

    References
    ---------
    .. [1] K. Enevoldsen et al. MMTEB: : Massive Multilingual Text Embedding Benchmark. In ICLR 2025
    """

    benchmark_name: str = "Multilingual"
    task_type: str = "Classification"
    max_n_docs: int | None = 8192

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Valid benchmark name
        assert self.benchmark_name in ["Multilingual", "Europe", "Indic", "eng, v2"], (
            f"Invalid benchmark name {self.benchmark_name}. See documentation for options."
        )

        # Valid task type
        assert self.task_type in [
            "Retrieval",
            "Classification",
            "Clustering",
            "STS",
            "PairClassification",
            "Reranking",
            "Summarization",
            "InstructionRetrieval",
        ], f"Invalid task type {self.task_type}. See documentation for options."


# ------------------------------------------------------------------------------
#   Benchmark
# ------------------------------------------------------------------------------
class MTEB:
    r"""
    MTEB benchmark from [1, 2]_.

    It consits of over 500 quality-controlled evaluation tasks across more than 250 languages.
    We focus on the 4 main benchmarks presented in [1]_, which are the "Multilingual", "Europe", "Indic"
    and "eng, v2". Other choices are possible, see the official website for more details. For efficiency reasons,
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

    Notes
    -----
    Official website: https://github.com/embeddings-benchmark/mteb?tab=readme-ov-file.

    References
    ----------
    .. [1] N. Muennighoff et al. MTEB: Massive Text Embedding Benchmark. In EACL 2023

    .. [2] K. Enevoldsen et al. MMTEB: : Massive Multilingual Text Embedding Benchmark. In ICLR 2025
    """

    def __init__(self, config: MTEBConfig):
        super().__init__()
        self.benchmark_name = config.benchmark_name
        self.task_type = config.task_type
        self.max_n_docs = config.max_n_docs

    def _standardize_task_format(self, tasks: list) -> list:
        r"""
        Ensure standard dataset format for MTEB tasks.

        There is a mismatch between the expected column names in MTEB evaluate function and the
        actual column names in some datasets hosted on HuggingFace which causes the following error:

        ```
        Original column names {'materia', 'summary'} not in the dataset.
        Current columns in the dataset: ['text', 'label']
        ```

        This error occurs for the ItaCaseholdClassification tasks. We bypass this issue by manually loading
        this dataset from Hugging Face and forcing the expected column schema (summary, materia, label).

        Parameters
        ----------
        tasks: list
            List of MTEB tasks.

        Returns
        -------
        standardized_tasks: list
            List of MTEB tasks with standardized format.

        Notes
        -----
        This patch solves an issue occuring with MTEB 2.10.4 and might become obsolete in future versions.
        """
        # Loop over tasks
        standardized_tasks = []
        for task in tasks:
            name = task.__class__.__name__
            if name in ["ItaCaseholdClassification"]:
                load_data = task.load_data
                path = task.metadata.dataset.get("path")

                # Wrap the load_data method
                @wraps(load_data)
                def patched_load_data(
                    *args, _func: Callable[..., Any] = load_data, _name: str = name, _path: str = path, **kwargs
                ):
                    logger.info(f"Manually loading and formatting dataset on task {_name}")

                    # Load dataset from Hugging Face
                    dataset = datasets.load_dataset(_path)

                    # Fix column names
                    for split in dataset.keys():
                        columns = dataset[split].column_names

                        # Map text column to summary column
                        if "text" in columns and "summary" not in columns:
                            dataset[split] = dataset[split].rename_column("text", "summary")

                        # Add an empty materia column
                        if "materia" not in dataset[split].column_names:
                            dataset[split] = dataset[split].add_column("materia", [""] * len(dataset[split]))

                    return dataset

                # Apply the patch to the instance
                task.load_data = patched_load_data

            # Add task to standardized tasks
            standardized_tasks.append(task)

        return standardized_tasks

    def evaluate(self, model: SentenceTransformer, cache_path: str) -> ModelResult:
        cache = mteb.ResultCache(cache_path=cache_path)
        return mteb.evaluate(model, tasks=self.tasks, cache=cache)

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

        # Ensure valid task format
        tasks = self._standardize_task_format(tasks)

        return tasks

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def domains(self) -> set[str]:
        return {domain for task in self.tasks for domain in (task.metadata.domains or [])}

    @property
    def languages(self) -> set[str]:
        return {lang for task in self.tasks for lang in (task.metadata.languages or [])}
