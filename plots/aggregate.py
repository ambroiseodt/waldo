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

# Variables

TRAINING_PARADIGMS = {
    "vit": "supervised",
    "dino": "ssl_distillation",
    "aim": "autoregressive",
    "clip": "contrastive",
    "siglip": "contrastive",
}


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
# Save results
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
                _, scores = data["scores"].popitem()
                accuracy = 100 * scores[0]["accuracy"]
                results.append(
                    {
                        "task_name": task_name,
                        "model_name": model_name,
                        "layer": layer,
                        "module": module,
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
        "accuracy",
    ]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RESULT_DIR / "mteb.csv"
    df.to_csv(save_path, columns=columns, index=False, na_rep="N/A", encoding="utf-8")
    logger.info(f"Results saved at {save_path}.")


def get_cls_results() -> None:
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
                    accuracy = 100 * data["acc_test"]
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
    save_path = RESULT_DIR / "cls.csv"
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
                        accuracy = 100 * data["acc_test"]
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
                    accuracy = 100 * data["acc_test"]
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


# ------------------------------------------------------------------------------
# Aggregate results
# ------------------------------------------------------------------------------
def analysis_cls_results() -> None:
    r"""
    Aggregate image classification results by recovering the best probing location,
    the corresponding accuracy, the baseline accuracy, defined as the probing accuracy of the rc2 module
    at the last layer, the absolute and relative gain of probing the best pair of layer-module instead
    of the baseline probing.
    """
    path = RESULT_DIR / "cls.csv"
    df = pd.read_csv(path)
    grouping_cols = [
        "dataset_name",
        "model_name",
        "model_size",
        "patch_size",
        "resolution",
    ]

    # Recover resoluts
    results = []
    for keys, group in tqdm(
        df.groupby(grouping_cols),
        desc="Building analysis table",
    ):
        dataset_name, model_name, model_size, patch_size, resolution = keys
        training_paradigm = next(
            (v for k, v in TRAINING_PARADIGMS.items() if k in str(model_name)),
            None,
        )
        n_layers = int(group["layer"].max()) + 1

        # Baseline probe
        last_layer = int(group["layer"].max())
        baseline_row = group[(group["layer"] == last_layer) & (group["module"] == "rc2")]
        baseline_accuracy = float(baseline_row["accuracy"].iloc[0])

        # Best probe
        best_idx = group["accuracy"].idxmax()
        best_row = group.loc[best_idx]
        best_layer = best_row["layer"]
        best_module = best_row["module"]
        best_accuracy = best_row["accuracy"]
        gain = best_accuracy - baseline_accuracy

        # Best layer with rc2 fixed
        rc2_df = group[group["module"] == "rc2"]
        best_layer_rc2_row = rc2_df.loc[rc2_df["accuracy"].idxmax()]
        best_layer_rc2 = best_layer_rc2_row["layer"]
        layer_accuracy = best_layer_rc2_row["accuracy"]
        gain_layer = layer_accuracy - baseline_accuracy

        # Best module at last layer
        last_layer_df = group[group["layer"] == last_layer]
        best_module_last_row = last_layer_df.loc[last_layer_df["accuracy"].idxmax()]
        best_module_last_layer = best_module_last_row["module"]
        module_accuracy = best_module_last_row["accuracy"]
        gain_module = module_accuracy - baseline_accuracy

        # Update results
        results.append(
            {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "model_size": model_size,
                "patch_size": patch_size,
                "resolution": resolution,
                "n_layers": n_layers,
                "training_paradigm": training_paradigm,
                "baseline_accuracy": baseline_accuracy,
                "best_layer": best_layer,
                "best_module": best_module,
                "best_accuracy": best_accuracy,
                "gain": gain,
                "best_layer_rc2": best_layer_rc2,
                "layer_accuracy": layer_accuracy,
                "gain_layer": gain_layer,
                "best_module_last_layer": best_module_last_layer,
                "module_accuracy": module_accuracy,
                "gain_module": gain_module,
            }
        )

    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values(by=["dataset_name", "model_name", "model_size"])
    save_path = RESULT_DIR / "analysis" / "cls.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(save_path, index=False, encoding="utf-8")
    logger.info(f"Saved table to {save_path}")


def analysis_domainnet_results() -> None:
    r"""
    Aggregate DomainNet results by recovering the best probing location,
    the corresponding accuracy, the baseline accuracy, defined as the probing accuracy of the rc2 module
    at the last layer, the absolute and relative gain of probing the best pair of layer-module instead
    of the baseline probing.
    """
    path = RESULT_DIR / "domainnet.csv"
    df = pd.read_csv(path)
    grouping_cols = [
        "domain",
        "model_name",
        "model_size",
        "patch_size",
        "resolution",
    ]

    # Recover resoluts
    results = []
    for keys, group in tqdm(
        df.groupby(grouping_cols),
        desc="Building analysis table",
    ):
        domain, model_name, model_size, patch_size, resolution = keys
        n_layers = int(group["layer"].max()) + 1
        training_paradigm = next(
            (v for k, v in TRAINING_PARADIGMS.items() if k in str(model_name)),
            None,
        )

        # Baseline probe
        last_layer = int(group["layer"].max())
        baseline_row = group[(group["layer"] == last_layer) & (group["module"] == "rc2")]
        baseline_accuracy = float(baseline_row["accuracy"].iloc[0])

        # Best probe
        best_idx = group["accuracy"].idxmax()
        best_row = group.loc[best_idx]
        best_layer = best_row["layer"]
        best_module = best_row["module"]
        best_accuracy = best_row["accuracy"]
        gain = best_accuracy - baseline_accuracy

        # Best layer with rc2 fixed
        rc2_df = group[group["module"] == "rc2"]
        best_layer_rc2_row = rc2_df.loc[rc2_df["accuracy"].idxmax()]
        best_layer_rc2 = best_layer_rc2_row["layer"]
        layer_accuracy = best_layer_rc2_row["accuracy"]
        gain_layer = layer_accuracy - baseline_accuracy

        # Best module at last layer
        last_layer_df = group[group["layer"] == last_layer]
        best_module_last_row = last_layer_df.loc[last_layer_df["accuracy"].idxmax()]
        best_module_last_layer = best_module_last_row["module"]
        module_accuracy = best_module_last_row["accuracy"]
        gain_module = module_accuracy - baseline_accuracy

        # Update results
        results.append(
            {
                "domain": domain,
                "model_name": model_name,
                "model_size": model_size,
                "patch_size": patch_size,
                "resolution": resolution,
                "n_layers": n_layers,
                "training_paradigm": training_paradigm,
                "baseline_accuracy": baseline_accuracy,
                "best_layer": best_layer,
                "best_module": best_module,
                "best_accuracy": best_accuracy,
                "gain": gain,
                "best_layer_rc2": best_layer_rc2,
                "layer_accuracy": layer_accuracy,
                "gain_layer": gain_layer,
                "best_module_last_layer": best_module_last_layer,
                "module_accuracy": module_accuracy,
                "gain_module": gain_module,
            }
        )

    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values(by=["domain", "model_name", "model_size"])
    save_path = RESULT_DIR / "analysis" / "domainnet.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(save_path, index=False, encoding="utf-8")
    logger.info(f"Saved table to {save_path}")


def analysis_cifar10_c_results() -> None:
    r"""
    Aggregate Cifar10-C results by recovering the best probing location,
    the corresponding accuracy, the baseline accuracy, defined as the probing accuracy of the rc2 module
    at the last layer, the absolute and relative gain of probing the best pair of layer-module instead
    of the baseline probing.
    """
    path = RESULT_DIR / "cifar10_c.csv"
    df = pd.read_csv(path)
    grouping_cols = [
        "corruption",
        "severity",
        "model_name",
        "model_size",
        "patch_size",
        "resolution",
    ]

    # Recover resoluts
    results = []
    for keys, group in tqdm(
        df.groupby(grouping_cols),
        desc="Building analysis table",
    ):
        corruption, severity, model_name, model_size, patch_size, resolution = keys
        n_layers = int(group["layer"].max()) + 1
        training_paradigm = next(
            (v for k, v in TRAINING_PARADIGMS.items() if k in str(model_name)),
            None,
        )

        # Baseline probe
        last_layer = int(group["layer"].max())
        baseline_row = group[(group["layer"] == last_layer) & (group["module"] == "rc2")]
        baseline_accuracy = float(baseline_row["accuracy"].iloc[0])

        # Best probe
        best_idx = group["accuracy"].idxmax()
        best_row = group.loc[best_idx]
        best_layer = best_row["layer"]
        best_module = best_row["module"]
        best_accuracy = best_row["accuracy"]
        gain = best_accuracy - baseline_accuracy

        # Best layer with rc2 fixed
        rc2_df = group[group["module"] == "rc2"]
        best_layer_rc2_row = rc2_df.loc[rc2_df["accuracy"].idxmax()]
        best_layer_rc2 = best_layer_rc2_row["layer"]
        layer_accuracy = best_layer_rc2_row["accuracy"]
        gain_layer = layer_accuracy - baseline_accuracy

        # Best module at last layer
        last_layer_df = group[group["layer"] == last_layer]
        best_module_last_row = last_layer_df.loc[last_layer_df["accuracy"].idxmax()]
        best_module_last_layer = best_module_last_row["module"]
        module_accuracy = best_module_last_row["accuracy"]
        gain_module = module_accuracy - baseline_accuracy

        # Update results
        results.append(
            {
                "corruption": corruption,
                "severity": severity,
                "model_name": model_name,
                "model_size": model_size,
                "patch_size": patch_size,
                "resolution": resolution,
                "n_layers": n_layers,
                "training_paradigm": training_paradigm,
                "baseline_accuracy": baseline_accuracy,
                "best_layer": best_layer,
                "best_module": best_module,
                "best_accuracy": best_accuracy,
                "gain": gain,
                "best_layer_rc2": best_layer_rc2,
                "layer_accuracy": layer_accuracy,
                "gain_layer": gain_layer,
                "best_module_last_layer": best_module_last_layer,
                "module_accuracy": module_accuracy,
                "gain_module": gain_module,
            }
        )

    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values(by=["corruption", "severity", "model_name", "model_size"])
    save_path = RESULT_DIR / "analysis" / "cifar10_c.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(save_path, index=False, encoding="utf-8")
    logger.info(f"Saved table to {save_path}")


def analysis_mteb_results() -> None:
    r"""
    Aggregate MTEB results by recovering the best probing location,
    the corresponding accuracy, the baseline accuracy, defined as the probing accuracy of the rc2 module
    at the last layer, the absolute and relative gain of probing the best pair of layer-module instead
    of the baseline probing.
    """

    # TEMP: some results are missing, so we bypass them in the results aggregation
    #####################
    # TEMP
    #####################
    MISSING_MODELS = ["gemma-4-E2B", "Qwen3-8B", "Qwen3-32B", "gemma-3-27b-pt"]
    #####################
    # TEMP
    #####################

    path = RESULT_DIR / "mteb.csv"
    df = pd.read_csv(path)
    grouping_cols = [
        "task_name",
        "model_name",
    ]

    # Recover resoluts
    results = []
    for keys, group in tqdm(
        df.groupby(grouping_cols),
        desc="Building analysis table",
    ):
        dataset_name, model_name = keys
        #####################
        # TEMP
        #####################
        if model_name in MISSING_MODELS:
            continue
        #####################
        # TEMP
        #####################

        n_layers = int(group["layer"].max()) + 1

        # Baseline probe
        last_layer = int(group["layer"].max())
        baseline_row = group[(group["layer"] == last_layer) & (group["module"] == "rc2")]
        baseline_accuracy = float(baseline_row["accuracy"].iloc[0])

        # Best probe
        best_idx = group["accuracy"].idxmax()
        best_row = group.loc[best_idx]
        best_layer = best_row["layer"]
        best_module = best_row["module"]
        best_accuracy = best_row["accuracy"]
        gain = best_accuracy - baseline_accuracy

        # Best layer with rc2 fixed
        rc2_df = group[group["module"] == "rc2"]
        best_layer_rc2_row = rc2_df.loc[rc2_df["accuracy"].idxmax()]
        best_layer_rc2 = best_layer_rc2_row["layer"]
        layer_accuracy = best_layer_rc2_row["accuracy"]
        gain_layer = layer_accuracy - baseline_accuracy

        # Best module at last layer
        last_layer_df = group[group["layer"] == last_layer]
        best_module_last_row = last_layer_df.loc[last_layer_df["accuracy"].idxmax()]
        best_module_last_layer = best_module_last_row["module"]
        module_accuracy = best_module_last_row["accuracy"]
        gain_module = module_accuracy - baseline_accuracy

        # Update results
        results.append(
            {
                "task_name": dataset_name,
                "model_name": model_name,
                "n_layers": n_layers,
                "baseline_accuracy": baseline_accuracy,
                "best_layer": best_layer,
                "best_module": best_module,
                "best_accuracy": best_accuracy,
                "gain": gain,
                "best_layer_rc2": best_layer_rc2,
                "layer_accuracy": layer_accuracy,
                "gain_layer": gain_layer,
                "best_module_last_layer": best_module_last_layer,
                "module_accuracy": module_accuracy,
                "gain_module": gain_module,
            }
        )

    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values(by=["task_name", "model_name"])
    save_path = RESULT_DIR / "analysis" / "mteb.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(save_path, index=False, encoding="utf-8")
    logger.info(f"Saved table to {save_path}")


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
            "cls": get_cls_results,
            "domainnet": get_domainnet_results,
            "cifar10_c": get_cifar10_c_results,
            "analysis_mteb": analysis_mteb_results,
            "analysis_cls": analysis_cls_results,
            "analysis_domainnet": analysis_domainnet_results,
            "analysis_cifar10_c": analysis_cifar10_c_results,
        }
    )


# %% CLI
if __name__ == "__main__":
    main()
# %%
