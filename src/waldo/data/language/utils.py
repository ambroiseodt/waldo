r"""
Utils functions to build the MTEB benchmark.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

import datasets

from ...utils import build_with_type_check
from .mteb import MTEB, MTEBConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
#   Standardize dataset format
# ------------------------------------------------------------------------------
def standardize_task_format(mteb_benchmark: MTEB) -> None:
    r"""
    Ensure standard dataset format for MTEB tasks.

    There is a mismatch between the expected column names in MTEB evaluate function and the
    actual column names in some datasets hosted on HuggingFace which causes the following error:

    ```
    Original column names {'materia', 'summary'} not in the dataset.
    Current columns in the dataset: ['text', 'label']
    ```

    We bypass this issue by checking for the expected format and if not found, we manually load data
    from Hugging Face and forces the expected column schema (summary, materia, label).

    Notes
    -----
    This patch is temporary and might become obsolete in newer versions of the MTEB library.
    """
    # Loop over tasks
    for task in mteb_benchmark.tasks:
        name = task.__class__.__name__
        load_data = task.load_data
        path = task.metadata.dataset.get("path")

        # Wrap the load_data method
        @wraps(load_data)
        def patched_load_data(
            *args, _func: Callable[..., Any] = load_data, _name: str = name, _path: str = path, **kwargs
        ):
            # Try loading data
            try:
                return _func(*args, **kwargs)

            # If there is a column name error, load manually and fix the issue
            except ValueError as error:
                prefix = "Original column names {'materia', 'summary'} not in the dataset."
                if prefix in str(error):
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

                # If it is an unrelated error, we raise it again
                else:
                    raise error

        # Apply the patch to the instance
        task.load_data = patched_load_data


# ------------------------------------------------------------------------------
#   Build MTEB
# ------------------------------------------------------------------------------
def build_mteb(
    config: dict[str, Any],
) -> MTEB:
    r"""
    Build dataloader with optimized parallelism settings.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration details.

    Returns
    -------
    benchmark: MTEB
        An instance of the specified MTEB benchmark.
    """

    config_obj = build_with_type_check(MTEBConfig, config)
    benchmark = MTEB(config_obj)
    standardize_task_format(benchmark)

    return benchmark
