r"""
Codebase to evaluate vision foundation models with linear probing on image classification.

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

from waldo.config import DEVICE, SAVING_DIR, set_seed
from waldo.data import build_loader
from waldo.models import build_model

logger = logging.getLogger("waldo")

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
    patch_size: int = 16
    resolution: int = 224

    # Data
    dataset_name: str = "cifar10"
    train_batch_size: int = 512
    test_batch_size: int = 512
    transform_type: str = "static"
    n_workers: int = 8

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
    model = build_model(config=model_config, device=config.device, return_config=False)
    logger.info("Done building model.")

    # ---------------------------------------------------------------------
    # Build dataloaders
    # ---------------------------------------------------------------------
    logger.info("Building dataloaders")
    train_loader_config = {
        "dataset_name": config.dataset_name,
        "batch_size": config.train_batch_size,
        "mode": "train",
        "resolution": config.resolution,
    }
    train_loader, n_classes = build_loader(config=train_loader_config, n_workers=config.n_workers, return_classes=True)
    test_loader_config = {
        "dataset_name": config.dataset_name,
        "batch_size": config.test_batch_size,
        "mode": "test",
        "resolution": config.resolution,
    }
    test_loader = build_loader(config=test_loader_config, n_workers=config.n_workers)
    logger.info("Done building dataloaders")

    # ---------------------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------------------

    # if config.cls_pooling:
    #     if config.finetuned:
    #         save_dir = PROBE_DIR / f"{config.log_dir}_cls_pooling"
    #     else:
    #         save_dir = PROBE_DIR / f"vit_{config.dataset_name}_seed_{exp_config['seed']}_pretrained_cls_pooling"
    # else:
    #     if config.finetuned:
    #         save_dir = PROBE_DIR / config.log_dir
    #     else:
    #         save_dir = PROBE_DIR / f"vit_{config.dataset_name}_seed_{exp_config['seed']}_pretrained"

    # save_dir.mkdir(exist_ok=True, parents=True)
    # with open(save_dir / "config.json", "w") as f:
    #     # JSON serializable alias
    #     exp_config = json_serializable(asdict(config))
    #     json.dump(exp_config, f, indent=4)

    # # ---------------------------------------------------------------------
    # # Linear probing
    # # ---------------------------------------------------------------------
    # metrics = run_linear_probing(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     cls_pooling=config.cls_pooling,
    #     device=config.device,
    #     seed=config.seed,
    # )
    # with open(save_dir / "linear_probing.json", "w") as f:
    #     json.dump(metrics, f, indent=4)

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
    python -m apps.language.eval config=apps/language/configs/your_config.yaml
    ``

    To overwrite some arguments via the cli, e.g., to set the device to cuda:0, run:

    ```bash
    python -m apps.language.eval config=apps/language/configs/your_config.yaml device=cuda:0
    ```
    """
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
