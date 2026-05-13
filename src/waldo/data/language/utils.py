r"""
Utils functions to build the MTEb benchmark.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
from functools import wraps
from typing import Any

import datasets

from ...utils import build_with_type_check
from .mteb import MTEB, MTEBConfig

logger = logging.getLogger(__name__)

import logging

logger = logging.getLogger(__name__)


def valid_task_format(mteb_benchmark):
    """
    Standardizes dataset format for MTEB tasks by bypassing broken load_data methods.

    This function intercepts the 'load_data' call. Instead of letting the
    broken MTEB task logic fail, it manually loads the dataset from Hugging Face
    and forces the expected column schema (summary, materia, label).
    """
    for task in mteb_benchmark.tasks:
        class_name = task.__class__.__name__

        # We wrap the load_data method
        original_load_data = task.load_data

        @wraps(original_load_data)
        def patched_load_data(*args, **kwargs):
            try:
                # First, try to load normally in case it's already fixed in library
                return original_load_data(*args, **kwargs)
            except ValueError as e:
                # If we hit the specific column name error, we manually intervene
                if "Original column names" in str(e):
                    logger.info(f"Intervening on {class_name}: Manually loading and formatting dataset.")

                    # 1. Get the dataset path from the task metadata
                    # Most tasks store the path in self.metadata.dataset['path']
                    # or it can be inferred from class name
                    path = "mteb/ItaCaseholdClassification"  # Targeted fix for the known broken path

                    # 2. Manual Load
                    ds = datasets.load_dataset(path)

                    # 3. Structural Fix
                    for split in ds.keys():
                        cols = ds[split].column_names
                        # Map text -> summary
                        if "text" in cols and "summary" not in cols:
                            ds[split] = ds[split].rename_column("text", "summary")
                        # Add dummy materia
                        if "materia" not in ds[split].column_names:
                            ds[split] = ds[split].add_column("materia", [""] * len(ds[split]))

                    return ds
                else:
                    # Reraise if it's an unrelated error
                    raise e

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
    valid_task_format(benchmark)

    return benchmark
