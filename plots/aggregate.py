r"""
Aggregate evaluation results from VFMs, VLMs and LLMs.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import json
import logging
import re

import fire
import pandas as pd
from tqdm import tqdm

from waldo.infra import RESULT_DIR, SAVING_DIR

# Logger setup
logger = logging.getLogger("waldo.aggregate")

# Paths
RUN_DIR = SAVING_DIR / "runs"


# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
def parse_model_config(name: str) -> tuple[str, int, str]:
    r"""
    Parse a model configuration name to recover model name, layer index, and module name.

    Parameters
    ----------
    name: str
        Model configuration name of the form "google/gemma-3-1b-pt_layer_0_module_act".

    Returns
    -------
    tuple[str, int, str]
        Tuple containing model name, layer index and module name.
    """
    pattern = re.compile(
        r"^"
        r"[^/]+"  # company name
        r"/"
        r"(.+?)"  # model_name
        r"_layer_"
        r"(\d+)"  # layer
        r"_module_"
        r"(.+)"  # module name
        r"$"
    )
    match = pattern.match(name)
    assert match is not None, f"Match not found. Check regex pattern and input {name}."
    model_name, layer, module = match.groups()

    return model_name, int(layer), module


def parse_result_config(name: str) -> tuple[int, str]:
    r"""
    Parse a result configuration name to recover layer index and module name.

    Parameters
    ----------
    name: str
        Result configuration name of the form "layer_0_module_act".

    Returns
    -------
    tuple[int, str]
        Tuple containing model name, layer index and module name.
    """
    pattern = re.compile(
        r"^"
        r"layer_"
        r"(\d+)"  # layer
        r"_module_"
        r"(.+?)"  # module name
        r".json"
    )
    match = pattern.match(name)
    assert match is not None, f"Match not found. Check regex pattern and input {name}."
    layer, module = match.groups()

    return int(layer), module


# ------------------------------------------------------------------------------
# MTEB results
# ------------------------------------------------------------------------------
def get_mteb_results() -> None:
    r"""
    Sweep over MTEB json files and aggregate results in a single csv.

    Notes
    -----
    MTEB does not modify the datasets structure is contains. As such,
    the evaluation is done on a test split when available. If only a train
    split is available, the logistic regression is fit and evaluated on independent
    samples to avoid data leakage. For more information, check out the official MTEB
    documentation.
    """

    # Check that path exists
    results_path = RUN_DIR / "mteb"
    if not results_path.exists():
        logger.info(f"Result directory '{results_path}' does not exist.")
        return

    # Compile results by looping over json files
    logger.info(f"Start recovering results from {results_path}.")
    results = []

    config_files = [f for f in results_path.glob("**/model_meta.json")]
    for config_path in tqdm(config_files, desc="Writing results", unit="runs"):
        # Recover model config
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_name, layer, module = parse_model_config(name=config["name"])

        # Recover results
        folder_path = config_path.parent
        results_files = [f for f in folder_path.glob("*.json") if f.name != "model_meta.json"]
        for json_path in results_files:
            with open(json_path, encoding="utf-8") as f:
                content = f.read().replace("NaN", "null")
                data = json.loads(content)
                task_name = data["task_name"]
                evaluation_time = data["evaluation_time"]
                key, scores = data["scores"].popitem()
                scores = scores[0]
                accuracies = [value["accuracy"] for value in scores["scores_per_experiment"]]

                # Sweep over experiments
                for id, accuracy in enumerate(accuracies):
                    results.append(
                        {
                            "task_name": task_name,
                            "model_name": model_name,
                            "layer": layer,
                            "module": module,
                            "evaluation_time": evaluation_time,
                            "experiment_id": id,
                            "accuracy": accuracy,
                        }
                    )

    # Aggregate results
    df = pd.DataFrame(results)
    column_order = ["task_name", "model_name", "layer", "module"]
    df = df.sort_values(by=column_order, ascending=True)
    columns = [
        "task_name",
        "model_name",
        "layer",
        "module",
        "evaluation_time",
        "experiment_id",
        "accuracy",
    ]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RESULT_DIR / "mteb.csv"
    df.to_csv(save_path, columns=columns, index=False, na_rep="N/A", encoding="utf-8")
    logger.info(f"Results saved at {save_path}.")


# ------------------------------------------------------------------------------
# Image classification results
# ------------------------------------------------------------------------------
def get_image_classification_results() -> None:
    r"""Sweep over image classification benchmarks json files and aggregate results in a single csv."""

    # Compile results by looping over datasets json files
    results = []
    dataset_names = [
        "aircraft",
        "cifar10",
        "cifar10_c-corruption-motion_blur-severity-5",
        "cifar100",
        "domainnet-sketch",
        "dtd",
        "eurosat",
        "flowers102",
        "food101",
        "pet",
    ]
    for dataset_name in dataset_names:
        # Check that path exists
        results_path = RUN_DIR / dataset_name
        if not results_path.exists():
            logger.info(f"Result directory '{results_path}' does not exist.")
            return

        logger.info(f"Start recovering results from {results_path}.")
        config_files = [f for f in results_path.glob("**/config.json")]
        for config_path in tqdm(config_files, desc=f"Writing results for {dataset_name}", unit="runs"):
            # Recover model config
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model_name = config["model_name"]
            model_size = config["model_size"]
            patch_size = config["patch_size"]
            resolution = config["resolution"]

            # Recover results
            folder_path = config_path.parent / "results"
            results_files = [f for f in folder_path.glob("*.json")]
            for json_path in results_files:
                # Recover layer index and module name
                layer, module = parse_result_config(name=json_path.name)
                with open(json_path, encoding="utf-8") as f:
                    content = f.read().replace("NaN", "null")
                    data = json.loads(content)
                    accuracy = data["acc_test"]
                    results.append(
                        {
                            "dataset_name": dataset_name,
                            "model_name": model_name,
                            "model_size": model_size,
                            "patch_size": patch_size,
                            "resolution": resolution,
                            "layer": layer,
                            "module": module,
                            "accuracy": accuracy,
                        }
                    )

    # Aggregate results
    df = pd.DataFrame(results)
    column_order = ["dataset_name", "model_name", "model_size", "patch_size", "resolution", "layer", "module"]
    df = df.sort_values(by=column_order, ascending=True)
    columns = [
        "dataset_name",
        "model_name",
        "model_size",
        "patch_size",
        "resolution",
        "layer",
        "module",
        "accuracy",
    ]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RESULT_DIR / "image_classification.csv"
    df.to_csv(save_path, columns=columns, index=False, na_rep="N/A", encoding="utf-8")
    logger.info(f"Results saved at {save_path}.")


def get_cifar10_c_results() -> None:
    r"""Sweep over Cifar10-C json files and aggregate results in a single csv."""

    # Check that path exists
    results_path = RUN_DIR / "cifar10_c"
    if not results_path.exists():
        logger.info(f"Result directory '{results_path}' does not exist.")
        return

    # Compile results by looping over datasets json files
    results = []
    logger.info(f"Start recovering results from {results_path}.")
    config_files = [f for f in results_path.glob("**/config.json")]
    for config_path in tqdm(config_files, desc="Writing results", unit="runs"):
        # Recover model config
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_name = config["model_name"]
        model_size = config["model_size"]
        patch_size = config["patch_size"]
        resolution = config["resolution"]
        corruptions = config["corruptions"]
        severities = config["severities"]

        # Recover results
        folder_path = config_path.parent / "results"
        for corruption in corruptions:
            for severity in severities:
                path = folder_path / f"cifar10_c-corruption-{corruption}-severity-{severity}"
                results_files = [f for f in path.glob("*.json")]
                for json_path in results_files:
                    # Recover layer index and module name
                    layer, module = parse_result_config(name=json_path.name)
                    with open(json_path, encoding="utf-8") as f:
                        content = f.read().replace("NaN", "null")
                        data = json.loads(content)
                        accuracy = data["acc_test"]
                        results.append(
                            {
                                "corruption": corruption,
                                "severity": severity,
                                "model_name": model_name,
                                "model_size": model_size,
                                "patch_size": patch_size,
                                "resolution": resolution,
                                "layer": layer,
                                "module": module,
                                "accuracy": accuracy,
                            }
                        )

    # Aggregate results
    df = pd.DataFrame(results)
    column_order = ["corruption", "severity", "model_name", "model_size", "patch_size", "resolution", "layer", "module"]
    df = df.sort_values(by=column_order, ascending=True)
    columns = [
        "corruption",
        "severity",
        "model_name",
        "model_size",
        "patch_size",
        "resolution",
        "layer",
        "module",
        "accuracy",
    ]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RESULT_DIR / "cifar10_c.csv"
    df.to_csv(save_path, columns=columns, index=False, na_rep="N/A", encoding="utf-8")
    logger.info(f"Results saved at {save_path}.")


def get_domainnet_results() -> None:
    r"""Sweep over DomainNets json files and aggregate results in a single csv."""

    # Check that path exists
    results_path = RUN_DIR / "domainnet"
    if not results_path.exists():
        logger.info(f"Result directory '{results_path}' does not exist.")
        return

    # Compile results by looping over datasets json files
    results = []
    logger.info(f"Start recovering results from {results_path}.")
    config_files = [f for f in results_path.glob("**/config.json")]
    for config_path in tqdm(config_files, desc="Writing results", unit="runs"):
        # Recover model config
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_name = config["model_name"]
        model_size = config["model_size"]
        patch_size = config["patch_size"]
        resolution = config["resolution"]
        domains = config["domains"]

        # Recover results
        folder_path = config_path.parent / "results"
        for domain in domains:
            path = folder_path / f"domainnet-{domain}"
            results_files = [f for f in path.glob("*.json")]
            for json_path in results_files:
                # Recover layer index and module name
                layer, module = parse_result_config(name=json_path.name)
                with open(json_path, encoding="utf-8") as f:
                    content = f.read().replace("NaN", "null")
                    data = json.loads(content)
                    accuracy = data["acc_test"]
                    results.append(
                        {
                            "domain": domain,
                            "model_name": model_name,
                            "model_size": model_size,
                            "patch_size": patch_size,
                            "resolution": resolution,
                            "layer": layer,
                            "module": module,
                            "accuracy": accuracy,
                        }
                    )

    # Aggregate results
    df = pd.DataFrame(results)
    column_order = ["domain", "model_name", "model_size", "patch_size", "resolution", "layer", "module"]
    df = df.sort_values(by=column_order, ascending=True)
    columns = [
        "domain",
        "model_name",
        "model_size",
        "patch_size",
        "resolution",
        "layer",
        "module",
        "accuracy",
    ]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RESULT_DIR / "domainnet.csv"
    df.to_csv(save_path, columns=columns, index=False, na_rep="N/A", encoding="utf-8")
    logger.info(f"Results saved at {save_path}.")


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(
        {
            "mteb": get_mteb_results,
            "cls": get_image_classification_results,
            "domain": get_domainnet_results,
            "cifar10_c": get_cifar10_c_results,
        }
    )


# %% CLI
if __name__ == "__main__":
    main()
# %%
