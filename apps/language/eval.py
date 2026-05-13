r"""
Codebase to evaluate large language models on the MTEB benchmark.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
from dataclasses import dataclass, field
from typing import cast

from omegaconf import OmegaConf

from waldo.config import DEVICE, SAVING_DIR, set_seed
from waldo.data import build_mteb
from waldo.models import build_model
from waldo.models.transformer_block import TRANSFORMER_MODULES

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
    model_name: str = "gemma3"
    model_size: str = "1b"

    # Probing
    layers: list[int | str] = field(default_factory=lambda: [-1])
    modules: list[str] = field(default_factory=lambda: ["rc2"])

    # Data
    benchmark_name: str = "Multilingual"
    task_type: str = "Classification"
    max_n_docs: int | None = 8192

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
    # Build MTEB benchmark
    # ---------------------------------------------------------------------
    logger.info("Building MTEB benchmark")
    mteb_config = {
        "benchmark_name": config.benchmark_name,
        "task_type": config.task_type,
        "max_n_docs": config.max_n_docs,
    }
    mteb_benchmark = build_mteb(config=mteb_config)
    logger.info(
        f"Done building MTEB benchmark with {mteb_benchmark.n_tasks} tasks, {len(mteb_benchmark.domains)} domains, "
        f"and {len(mteb_benchmark.languages)} languages"
    )

    # --- DIAGNOSTIC CHECK ---
    print("\n" + "=" * 50)
    print("RUNNING POST-PATCH DIAGNOSTICS")
    print("=" * 50)

    for task in mteb_benchmark.tasks:
        # Use __class__.__name__ since 'description' attribute is missing
        class_name = task.__class__.__name__

        if "ItaCaseholdClassification" in class_name:
            print(f"Checking Task: {class_name}")
            try:
                # Trigger the patched load_data
                test_ds = task.load_data()
                # Get first available split
                s_name = list(test_ds.keys())[0]
                cols = test_ds[s_name].column_names
                print(f"  Split: {s_name}")
                print(f"  Columns Found: {cols}")

                if "summary" in cols and "materia" in cols:
                    print("  RESULT: SUCCESS - Columns are mapped correctly.")
                else:
                    print("  RESULT: FAILED - Columns are still incorrect.")
            except Exception as e:
                print(f"  RESULT: ERROR during load - {e}")

    print("=" * 50 + "\n")

    # ---------------------------------------------------------------------
    # Run evaluation  with the defined probes to obtain the sentence embedding
    # ---------------------------------------------------------------------
    cache_path = RUN_DIR / "mteb" / model.model_name

    # Recover layers and modules with a condition to select all of them
    layers = config.layers
    if layers == ["all"]:
        layers = range(model.n_layers)
    modules = config.modules
    if modules == ["all"]:
        modules = TRANSFORMER_MODULES

    # Loop over layers and modules to probe to embed sentences
    for layer in layers:
        for module in modules:
            logging.info(f"Probing layer {layer} and module {module} for hidden state extraction.")
            model.set_probing(layer=layer, module=module)
            mteb_benchmark.evaluate(model=model, cache_path=cache_path)

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
