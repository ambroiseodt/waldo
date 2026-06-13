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

import fire
import pandas as pd
from waldo.infra import SAVING_DIR, RESULT_DIR

# Logger setup
logger = logging.getLogger("waldo.aggregate")

# Paths
RUN_DIR = SAVING_DIR / "runs"


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
    logger.info(f"Starting dynamic sweep of: {results_path.resolve()}")
    results = []
    for config_path in results_path.glob("**/model_config.json"):
        folder_path = config_path.parent

        # Recover model config
        with open(config_path, encoding="utf-8") as f:
            config = json.loads(config_path)
            # TODO: recover layer and module
            model_name = config["model_name"]
            model_size = ""
            layer = 0
            module = ""

        # Recover results
        results_files = [f for f in folder_path.glob("*.json") if f.name != "model_config.json"]
        for json_path in results_files:
            with open(json_path, encoding="utf-8") as f:
                content = f.read().replace("NaN", "null")
                data = json.loads(content)
                task_name = data["task_name"]
                evaluation_time = data["evaluation_time"]
                scores = data["scores"]
                if "test" in scores:
                    scores = scores["test"][0]
                else:
                    scores = scores["train"][0]
                accuracies = [value["accuracy"] for value in scores["score_per_experiment"]]

                # Sweep over experiments
                for idx, accuracy in accuracies:
                    results.append(
                        {
                            "model_name": model_name,
                            "model_size": model_size,
                            "layer": layer,
                            "module": module,
                            "task_name": task_name,
                            "experiment_index": idx,
                            "accuracy": accuracy,
                            "evaluation_time": evaluation_time,
                        }
                    )

    # Aggregate results
    df = pd.DataFrame(results)
    column_order = [
        "model_name",
        "model_size",
        "layer",
        "module",
        "task",
        "experiment_index",
        "accuracy",
        "evaluation_time",
    ]
    df = df[column_order]
    save_path = RESULT_DIR / "mteb.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)
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
