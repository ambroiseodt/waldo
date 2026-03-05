r"""
Codebase to preprocess the MTEB language benchmark.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from typing import Any

import mteb
from torch.utils.data import DataLoader

names = ["Multilingual", "Europe", "Indic", "eng, v2"]
comp = 0
for benchmark_name in names:
    # Get the full English benchmark
    benchmark = mteb.get_benchmark(f"MTEB({benchmark_name})")

    # Filter to only retrieval tasks
    classification_tasks = mteb.filter_tasks(benchmark, task_types=["Classification"])
    comp += len(classification_tasks)
    print(f"Found {len(classification_tasks)} classification tasks")
print(comp)


def build_taks(
    config: dict[str, Any],
    num_workers: int = 8,
    return_n_classes: bool = False,
) -> DataLoader | tuple[DataLoader, int]:
    r"""
    Build dataloader with optimized parallelism settings.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration details.
    num_workers: int, default=8
        The number of workers to use for data loading.
    return_n_classes: bool, default=False
        Whether to return the number of classes in the dataset.

    Returns
    -------
    loader: DataLoader | tuple[DataLoader, int]
        An instance of the specified dataloader implementation.

    Notes
    -----
    Loading data can be slow if the num_workers variable is not properly set. The default value is set to 8 based on
    empirical benchmarking on our datasets. Despite light augmentations, multiple workers improve throughput due to
    I/O parallelism when reading from disk. OMP_NUM_THREADS is set to 1 per worker via worker_init_fn to prevent thread
    contention: if num_workers > 0, each worker spawns its own OpenMP threads, leading to num_workers x OMP_NUM_THREADS
    competing threads. Restricting workers to a single thread avoids this while leaving the main process thread budget
    untouched for model training. This follows the protocol of https://github.com/facebookresearch/lingua.
    """

    # Argument parsing
    batch_size = config.pop("batch_size", 128)
    resolution = config.pop("resolution", 224)
    transform_type = config.pop("transform_type", "train")

    # Set data transform
    config = config | dict(transform=build_transform(resolution=resolution, transform_type=transform_type))

    # Create dataset
    dataset, n_classes = build_dataset(config)

    # Create dataloader
    shuffle = config["mode"] == "train"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=_worker_init_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Recover number of classes
    if return_n_classes:
        return loader, n_classes

    return loader
