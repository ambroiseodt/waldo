r"""
Multilingual Text Embedding Benchmark.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import dataclass

import mteb


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
    max_n_documents: int | None
        Maximul number of documents to use. This allows us to filter out large tasks that are too
        costly to evaluate on. If None, all the tasks are used.

    Reference
    ---------
    .. [1] K. Enevoldsen et al. MMTEB: : Massive Multilingual Text Embedding Benchmark. In ICLR 2025
    """

    benchmark_name: str = "Multilingual"
    task_type: str = "Classification"
    max_n_documents: int | None = None

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


class MTEB:
    r"""
    MTEB benchmark from [1, 2]_.

    It consits of over 500 quality-controlled evaluation tasks across more than 250 languages.

    Parameters
    ----------
    config: configuration class with
        benchmark_name: str, default="Multilingual"
            Name of the benchmark.
        task_type: str, default="Classification"
            Type of task.
        max_n_documents: int | None, default=None
            Maximul number of documents to use.

    Notes
    ----------
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
        self.max_n_documents = config.max_n_documents

    def get_tasks(self) -> list:
        r"""
        Get the tasks of the benchmark.

        Returns
        -------
        tasks: list
            List of tasks.
        """
        # Get the full English benchmark
        benchmark = mteb.get_benchmark(f"MTEB({self.benchmark_name})")

        # Filter to only retrieval tasks
        tasks = mteb.filter_tasks(benchmark, task_types=[self.task_type])

        # Filter out large tasks if max_n_documents is set
        if self.max_n_documents is not None:
            tasks = [
                task for task in tasks if task.metadata_dict.get("n_documents", float("inf")) <= self.max_n_documents
            ]

        return tasks
