r"""
Setup functions to launch experiments.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
import sys


def setup_logging() -> None:
    r"""
    Setup logger and standardizes the console output by silencing redundant library warnings.

    This suppresses:
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


