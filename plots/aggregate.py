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
from waldo.infra import SAVING_DIR, RESULT_DIR

# Logger setup
logger = logging.getLogger("waldo.aggregate")

# Paths
RUN_DIR = SAVING_DIR / "runs"


# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
def parse_model_config(name: str) -> tuple[str, int, str] | None:
    r"""
    Parse a model configuration name to recover model name, layer index, and module name.

    Parameters
    ----------
    config_string: str
        Configuration name of the form "google/gemma-3-1b-pt_layer_0_module_act".

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
    if not match:
        return None

    model_name, layer, module = match.groups()

    return model_name, int(layer), module


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

    for i, config_path in enumerate(results_path.glob("**/model_meta.json")):
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
                avg_accuracy = scores["accuracy"]
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
                            "avg_accuracy": avg_accuracy,
                            "experiment_id": id,
                            "accuracy": accuracy,
                        }
                    )
        # Logger
        if i % 100:
            logger.info(f"Recover results for run number {i}.")

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
        "avg_accuracy",
        "experiment_id",
        "accuracy",
    ]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RESULT_DIR / "mteb.csv"
    df.to_csv(save_path, columns=columns, index=False, na_rep="N/A", encoding="utf-8")
    logger.info(f"Results saved at {save_path}.")


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire({"mteb": get_mteb_results})


# %% CLI
if __name__ == "__main__":
    main()
# %%
