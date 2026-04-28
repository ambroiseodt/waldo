r"""
Orchestrator manager to monitor the logging and garbage collection.

See also
--------
logger.py
utility.py

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import SAVING_DIR
from ..utils import build_with_type_check, json_serializable
from .logger import LoggerConfig
from .utility import UtilityConfig

logger = logging.getLogger("vitef")

# Paths
RUN_DIR = SAVING_DIR / "runs"


# ------------------------------------------------------------------------------
# Training Orchestrator
# ------------------------------------------------------------------------------


@dataclass
class OrchestratorConfig:
    r"""
    Orchestrator to handle checkpointer, logger and utility.

    Parameters
    ----------
    log_dir: str
        Path to the root directory of the experiment folder.
    overwrite: bool
        Whether to overwrite the experiment folder.
    config_file: dict[str, Any]
        Dictionary containing the configuration details of the experiment.
    logging_period: int
        Number of updates between each logging.
    logging_level: str
        Logging level.
    seed: int
        Seed for reproducibility.
    utility_period: int
        Number of updates between each garbage collection.
    """

    log_dir: Path = Path("")
    overwrite: bool = False

    # Experiment configuration
    config_file: dict[str, Any] | None = None

    # Logger
    logging_period: int = 0
    logging_level: str = "INFO"

    # Utility
    seed: int = 42
    utility_period: int = 1000

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self) -> None:
        assert self.log_dir, "log_dir should be specified."

        # Logging directory
        self.log_dir = RUN_DIR / self.log_dir

        # Create or overwrite logging directory
        if self.log_dir.exists() and self.overwrite:
            confirm = input(f"Do you want to permanently delete the directory '{self.log_dir}' (Yes/No)? ")
            if confirm.upper().startswith("Y"):
                shutil.rmtree(self.log_dir)
                logger.info(f"Directory '{self.log_dir}' has been deleted.")
            else:
                logger.info(
                    (
                        "Operation cancelled. Start again if it was a mistake.",
                        f"Otherwise, we recommend setting overwrite to {not self.overwrite}.",
                    )
                )
                sys.exit(0)
        self.log_dir.mkdir(parents=True, exist_ok=True)


def build_orchestrator(config: dict[str, Any]) -> dict[str, Any]:
    r"""
    Build the orchestrator to handle checkpoints, logging, reproducibility and garbage collection.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration details.

    Returns
    -------
    orchestrator: dict
        Dictionary of config files.
    """
    config_obj = build_with_type_check(OrchestratorConfig, config)

    # Save experiment configuration
    with open(config_obj.log_dir / "config.json", "w") as f:
        # JSON serializable alias
        config_file = json_serializable(config_obj.config_file)
        json.dump(config_file, f, indent=4)

    # Logger
    config["period"] = config_obj.logging_period
    config["level"] = config_obj.logging_level
    config["stdout_path"] = str(config_obj.log_dir / "logs")
    config["metric_path"] = str(config_obj.log_dir / "metrics")
    logger_config = build_with_type_check(LoggerConfig, config)

    # Utility
    config["seed"] = config_obj.seed
    config["period"] = config_obj.utility_period
    utility_config = build_with_type_check(UtilityConfig, config)

    # Orchestrator
    orchestrator = {"logger": logger_config, "utility": utility_config}

    return orchestrator
