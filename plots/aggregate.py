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

from waldo.infra import SAVING_DIR

# Logger setup
logger = logging.getLogger("waldo.aggregate")

# Paths
RUN_DIR = SAVING_DIR / "runs"


# ------------------------------------------------------------------------------
# MTEB results
# ------------------------------------------------------------------------------
def get_mteb_results() -> None:
    """Dynamically sweeps the MTEB directory tree and builds a single master CSV

    by parsing files directly within a unified execution loop.
    """

    # Check that path exists
    results_path = RUN_DIR / "mteb"
    if not results_path.exists():
        logger.info(f"Result directory '{results_path}' does not exist.")
        return

    # Compile results by looping over json files
    logger.info(f"Starting dynamic sweep of: {results_path.resolve()}")
    for json_path in results_path.glob("**/*.json"):
        with open(json_path, encoding="utf-8") as f:
            content = f.read().replace("NaN", "null")
            data = json.loads(content)
            task_name = data
            print(json_path)
            print(data.keys())
            return

    #     # Ignore metadata/config files
    #     if json_path.name in ["config.json", "metadata.json"]:
    #         continue

    #     try:
    #         # 1. Decompose the path to recover structural configuration traits
    #         # Example: ../results/language/bert/large/layer_12/encoder/results/main/task.json
    #         relative_parts = json_path.relative_to(MTEB_ROOT_DIR).parts

    #         model_name = relative_parts[0] if len(relative_parts) > 0 else "unknown"
    #         model_size = relative_parts[1] if len(relative_parts) > 1 else "unknown"
    #         layer = relative_parts[2] if len(relative_parts) > 2 else "unknown"
    #         module = relative_parts[3] if len(relative_parts) > 3 else "unknown"

    #         # 2. Open and load the JSON payload directly
    #         with open(json_path, encoding="utf-8") as f:
    #             content = f.read().replace("NaN", "null")
    #             data = json.loads(content)

    #         task_name = data.get("task_name")
    #         eval_time = data.get("evaluation_time")

    #         test_scores = data.get("scores", {}).get("test", [])
    #         if not test_scores:
    #             continue

    #         experiments = test_scores[0].get("scores_per_experiment", [])

    #         # 3. Dynamic online loop over the multiple runs
    #         for idx, exp in enumerate(experiments):
    #             compiled_rows.append(
    #                 {
    #                     "model_name": model_name,
    #                     "model_size": model_size,
    #                     "layer": layer,
    #                     "module": module,
    #                     "task": task_name,
    #                     "experiment_index": idx,
    #                     "accuracy": exp.get("accuracy"),
    #                     "evaluation_time": eval_time,
    #                 }
    #             )

    #     except (json.JSONDecodeError, KeyError, IndexError, FileNotFoundError) as e:
    #         # Skip corrupt or half-written log checkpoints without crashing the loop
    #         print(f"  [Warning] Skipping unreadable file {json_path.name}: {e}")
    #         continue

    # # 4. Save compiled dataset to a single flat CSV
    # if compiled_rows:
    #     df = pd.DataFrame(compiled_rows)

    #     column_order = [
    #         "model_name",
    #         "model_size",
    #         "layer",
    #         "module",
    #         "task",
    #         "experiment_index",
    #         "accuracy",
    #         "evaluation_time",
    #     ]
    #     df = df[column_order]

    #     OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    #     df.to_csv(OUTPUT_CSV, index=False)
    #     print(f"Success! Captured {len(df)} lines inside -> {OUTPUT_CSV}")
    # else:
    #     print("Aggregation finished: No valid MTEB target JSON records found.")


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
