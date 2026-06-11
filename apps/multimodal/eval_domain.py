r"""
Codebase to evaluate vision language models via linear probing for domain generalization.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import cast

from omegaconf import OmegaConf

from waldo.data import build_loader
from waldo.eval import build_linear_probing
from waldo.infra import DEVICE, SAVING_DIR, json_serializable, set_seed
from waldo.models import build_model

from ..setup import setup_logging

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
    model_name: str = "clip"
    model_size: str = "base"
    patch_size: int = 16
    resolution: int = 224

    # Data
    domains: list = field(default_factory=lambda: ["clipart", "infograph", "painting", "quickdraw", "sketch"])
    batch_size: int = 512

    # Linear probing
    tol: float = 1e-4
    max_iter: int = 1000

    # Logger
    overwrite: bool = False

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

    Returns
    -------
    cache_path: Path
        Path where the results are saved.
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
        "patch_size": config.patch_size,
        "resolution": config.resolution,
        "device": config.device,
    }
    model = build_model(config=model_config, return_config=False)
    logger.info("Done building model.")

    # ---------------------------------------------------------------------
    # Build dataloaders: train on DomainNet-Real and test on another domain
    # ---------------------------------------------------------------------
    logger.info("Building train and test dataloaders")
    loader_config = {
        "batch_size": config.batch_size,
        "resolution": config.resolution,
        "transform_type": "static",
    }
    train_loader_config = loader_config | {"dataset_name": "domainnet-real", "mode": "train"}
    train_loader, n_classes = build_loader(config=train_loader_config, return_n_classes=True)
    test_loader = {}
    for domain in config.domains:
        dataset_name = f"domainnet-{domain}"
        test_loader_config = loader_config | {"dataset_name": dataset_name, "mode": "test"}
        test_loader[dataset_name] = build_loader(config=test_loader_config)
    logger.info(f"Done building dataloaders for {len(test_loader)} DomainNet variants {n_classes} classes.")

    # ---------------------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------------------
    cache_path = RUN_DIR / "domainnet" / model.model_name
    cache_path.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    with open(cache_path / "config.json", "w") as f:
        json.dump(json_serializable(asdict(config)), f, indent=4)

    # Launch linear probing
    logger.info("Running linear probing on DomainNet.")
    linear_probing_config = {
        "tol": config.tol,
        "max_iter": config.max_iter,
        "device": config.device,
        "seed": config.seed,
    }
    linear_probing = build_linear_probing(config=linear_probing_config)
    linear_probing.evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        cache_path=cache_path,
        overwrite=config.overwrite,
    )


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
    python -m apps.multimodal.eval_domain config=apps/multimodal/configs/your_config.yaml
    ``

    To overwrite some arguments via the cli, e.g., to set the device to cuda:0, run:

    ```bash
    python -m apps.multimodal.eval_domain config=apps/multimodal/configs/your_config.yaml device=cuda:0
    ```
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
