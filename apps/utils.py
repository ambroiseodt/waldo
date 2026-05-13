r"""
Utils functions to launch linear probing experiments.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
import os
import sys
import warnings

from sklearn.exceptions import ConvergenceWarning


def setup_logging() -> None:
    r"""
    Setup logger and standardizes the console output by silencing redundant library warnings.

    This suppresses:
    - Sklearn ConvergenceWarnings (common in MTEB classification).
    - ResourceWarnings/FutureWarnings from libraries like pandas and datasets.
    - Verbose INFO/WARNING logs from Transformers and LiteLLM.
    """
    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Silence sklearn built-in warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    # Set noisy library logger level to ERROR
    noisy_loggers = ["transformers", "datasets", "httpx", "filelock", "openai", "litellm"]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Silence Tokenizer parallelism warning if using many CPUs
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
