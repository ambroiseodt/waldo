r"""
Analyze evaluation results from VFMs, VLMs and LLMs.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from waldo.infra import FIGURE_DIR, RESULT_DIR

logger = logging.getLogger("waldo.analysis")

# Variables
VFMs = ["aim", "dino", "vit"]
VLMs = ["clip", "siglip"]
LLMs = ["gemma", "llama", "qwen"]


# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
def save_plot(figname: str, format: str = "pdf", dpi: int = 100) -> None:
    """Save figure in pdf format."""
    figure_path = FIGURE_DIR
    if not figure_path.exists():
        figure_path.mkdir(parents=True, exist_ok=True)
    save_dir = figure_path / f"{figname}.{format}"
    plt.savefig(save_dir, format=format, bbox_inches="tight", dpi=dpi)


# ------------------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------------------
def plot_gap_distribution_vfm(save: bool = False) -> None:
    # Recover results
    path = RESULT_DIR / "analysis" / "domainnet.csv"
    df = pd.read_csv(path)
    pattern = "|".join(VLMs)
    df = df[~df["model_name"].str.contains(pattern, case=False, na=False)]
    pattern = "|".join(["1B", "3B", "vit7b16", "giant", "huge", "large"])
    df = df[~df["model_size"].str.contains(pattern, case=False, na=False)]
    # df = df[~df["severity"].isin([1, 3])]
    pattern = "|".join(["infograph", "quickdraw"])
    df = df[~df["domain"].str.contains(pattern, case=False, na=False)]
    plot_df = pd.DataFrame(
        {
            "gap": pd.concat(
                [df["gap_best_layer_rc2"], df["gap_best_module_last_layer"], df["gap"]], ignore_index=True
            ),
            "type": (["best layer"] * len(df) + ["best module"] * len(df) + ["best location"] * len(df)),
        }
    )
    plot_df["gap"] = 100 * plot_df["gap"]
    plt.figure(figsize=(6, 5))

    sns.boxplot(
        data=plot_df,
        x="type",
        y="gap",
        showfliers=False,  # IMPORTANT: removes outlier clutter
    )

    # sns.stripplot(data=plot_df, x="type", y="gap", color="black", alpha=0.25, size=2)

    plt.axhline(0, color="gray", linestyle="--", linewidth=1)

    plt.xlabel("")
    plt.ylabel("Gain over standard probing")
    plt.title("Probe-location optimization improves performance")

    plt.tight_layout()
    if save:
        figname = "gap_distribution"
        save_plot(figname=figname)


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire({"gap": plot_gap_distribution_vfm})


# %% CLI
if __name__ == "__main__":
    main()
# %%
