r"""
Codebase to evaluate vision foundation models via linear probing on image classification tasks.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
from dataclasses import dataclass
from typing import cast

from omegaconf import OmegaConf

from waldo.data import build_loader
from waldo.eval.linear import linear_probing
from waldo.infra.config import DEVICE, SAVING_DIR, set_seed
from waldo.models import build_model

from ..utils import setup_logging

# Logger setup
logger = logging.getLogger("waldo.eval")

# Paths
RUN_DIR = SAVING_DIR / "runs"


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
@dataclass
class EvaluationConfig:
    r"""Evaluation config."""

    # Model
    model_name: str = "vit"
    model_size: str = "base"
    patch_suze: int = 16
    resolution: int = 224

    # Data
    dataset_name: str = "cifar10"
    batch_size: int = 512

    # Device
    device: str = DEVICE

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        r"""
        Set up the seed.
        """
        if self.seed is None:
            self.seed = 42


# ------------------------------------------------------------------------------
# Evaluation via linear probing
# ------------------------------------------------------------------------------
def eval(config: EvaluationConfig) -> None:
    r"""
    Run the evaluation from the config file and save the results.

    Parameters
    ----------
    config: EvaluationConfig
        Configuration file to setup the model and the benchmark.
    """
    # ---------------------------------------------------------------------
    # Set seed for reproducibility
    # ---------------------------------------------------------------------
    set_seed(config.seed)

    # ---------------------------------------------------------------------
    # Build model
    # ---------------------------------------------------------------------
    logger.info("Building model.")
    model_config = {
        "model_name": config.model_name,
        "model_size": config.model_size,
        "device": config.device,
    }
    model = build_model(config=model_config, return_config=False)
    logger.info("Done building model.")

    # ---------------------------------------------------------------------
    # Build dataloaders
    # ---------------------------------------------------------------------
    logger.info("Building train and test dataloaders")
    train_loader_config = {
        "dataset_name": config.dataset_name,
        "batch_size": config.batch_size,
        "mode": "train",
        "resolution": config.resolution,
        "transform_type": "static",
    }
    train_loader, n_classes = build_loader(config=train_loader_config, return_n_classes=True)
    test_loader_config = {
        "dataset_name": config.dataset_name,
        "batch_size": config.batch_size,
        "mode": "train",
        "resolution": config.resolution,
        "transform_type": "static",
    }
    test_loader = build_loader(config=test_loader_config)
    logger.info(f"Done building dataloaders for {config.dataset_name} with {n_classes} classes.")

    # ---------------------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------------------
    metrics = linear_probing()

    logger.info("Evaluation done.")


# %% Main
def main() -> None:
    r"""
    Launch an evaluation job from a configuration file specified by cli argument.

    The command line interface here uses OmegaConf
    (see https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments).

    The behavior here is as follows:
    1. We instantiate the config file with its default values,
    2. We override those default values with the ones in the provided config file,
    3. We override the result with the additional arguments provided through command line.

    Usage:
    To launch a training according to your_config.yaml file, run:

    ```bash
    python -m apps.vision.eval config=apps/vision/configs/your_config.yaml
    ``

    To overwrite some arguments via the cli, e.g., to set the device to cuda:0, run:

    ```bash
    python -m apps.vision.eval config=apps/language/configs/your_config.yaml device=cuda:0
    ```vision
    """
    # Logger setup
    setup_logging()

    # Recover config from CLI
    cli_args = OmegaConf.from_cli()
    file_config = OmegaConf.load(cli_args.config)

    # Remove 'config' attribute from config as the underlying dataclass does not have it
    del cli_args.config

    # Recover default config and merge all of them
    default_config = OmegaConf.structured(EvaluationConfig())
    merged_config = OmegaConf.merge(default_config, file_config, cli_args)
    config = cast(EvaluationConfig, OmegaConf.to_object(merged_config))

    # Launch linear probing
    eval(config=config)


# %% CLI
if __name__ == "__main__":
    main()
# %%


# ------------------------------------------------------------------------------
# Main Orchestration Loop
# ------------------------------------------------------------------------------
def linear_probing(config: LinearProbingConfig) -> None:
    r"""Main execution loop for model tracking, feature collection, and metric packaging."""

    # Set execution seed explicitly
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Setup hardware allocation dummy params (simulate internal project dictionary)
    model_config = {
        "model_name": config.model_name,
        "model_size": config.model_size,
        "pretrained": not config.finetuned,
    }

    # 1. Instantiate the foundation model
    logger.info("Initializing Large Vision Foundation Model...")
    model = build_model(config=model_config, device=config.device, return_config=False)

    # 2. Build explicit dataloaders passing the worker configurations
    logger.info("Assembling dataset streams...")
    loader_config = {
        "dataset_name": config.dataset_name,
        "batch_size": config.batch_size,
        "val_batch_size": config.val_batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
        "persistent_workers": True,
    }

    train_loader, _, _ = build_train_val_loader(
        config=loader_config, train_size=config.train_size, return_n_classes=True
    )

    test_loader_config = {
        "dataset_name": config.dataset_name,
        "batch_size": config.test_batch_size,
        "mode": "test",
        "num_workers": config.num_workers,
        "pin_memory": True,
    }
    test_loader = build_loader(config=test_loader_config, drop_last=False)

    # 3. Execute probing script with microsecond tracking
    start_time = time.perf_counter()

    metrics = run_linear_probing(model=model, train_loader=train_loader, test_loader=test_loader, config=config)

    elapsed_time = time.perf_counter() - start_time

    # 4. Save results to the MTEB structured folder hierarchy
    save_mteb_results(metrics=metrics, config=config, elapsed_time=elapsed_time)
