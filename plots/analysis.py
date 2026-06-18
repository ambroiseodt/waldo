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
import numpy as np
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
    path = RESULT_DIR / "analysis" / "image_classification.csv"
    df = pd.read_csv(path)
    pattern = "|".join(VLMs)
    df = df[~df["model_name"].str.contains(pattern, case=False, na=False)]
    # pattern = "|".join(["dinov3"])
    # df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    # pattern = "|".join(["vits"])
    # df = df[df["model_size"].str.contains(pattern, case=False, na=False)]
    # pattern = "|".join(["1B", "3B", "vit7b16", "giant", "huge", "large"])
    # df = df[~df["model_size"].str.contains(pattern, case=False, na=False)]
    # df = df[df["severity"].isin([5])]
    # pattern = "|".join(["quickdraw"])
    # df = df[~df["domain"].str.contains(pattern, case=False, na=False)]
    plot_df = pd.DataFrame(
        {
            "gap": pd.concat(
                [
                    df["gap_best_layer_rc2"],
                    df["gap_best_module_last_layer"],
                    df["gap"],
                    # df["gap"] - df["gap_best_layer_rc2"],
                    # df["gap"] - df["gap_best_module_last_layer"],
                ],
                ignore_index=True,
            ),
            "type": (["best layer"] * len(df) + ["best module"] * len(df) + ["best location"] * len(df)),
            # "type": (["best layer"] * len(df) + ["best module"] * len(df)),
        }
    )
    plot_df["gap"] = 100 * plot_df["gap"]
    plt.figure(figsize=(6, 5))

    sns.boxplot(
        data=plot_df,
        x="type",
        y="gap",
        showfliers=False,  # IMPORTANT: removes outlier clutter
        showmeans=True,
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


def plot_cdf(ax, df, title):
    for col, label in [("gap_best_layer_rc2", "Layer"), ("gap_best_module_last_layer", "Module"), ("gap", "Location")]:
        x = np.sort(df[col].values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, 1 - y, linewidth=2, label=label)

    ax.set_title(title)
    ax.set_xlabel("Gain")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.3)
    ax.legend()


def plot_paired(ax, df, title):
    data = pd.DataFrame(
        {"Layer": df["gap_best_layer_rc2"], "Module": df["gap_best_module_last_layer"], "Location": df["gap"]}
    )

    melted = data.melt(var_name="Method", value_name="Gain")

    sns.barplot(
        data=melted,
        x="Method",
        y="Gain",
        ax=ax,
        errorbar=("ci", 95),  # <-- THIS adds std bars
    )

    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Gain (mean ± std)")
    ax.grid(axis="y", alpha=0.3)


def plot_all(save: bool = False):

    path = RESULT_DIR / "analysis"
    df_id = pd.read_csv(path / "image_classification.csv")
    df_domain = pd.read_csv(path / "domainnet.csv")
    df_cifar = pd.read_csv(path / "cifar10_c.csv")
    pattern = "|".join(VLMs)
    df_id = 100 * df_id[~df_id["model_name"].str.contains(pattern, case=False, na=False)]
    df_domain = df_domain[~df_domain["model_name"].str.contains(pattern, case=False, na=False)]
    df_cifar = df_cifar[~df_cifar["model_name"].str.contains(pattern, case=False, na=False)]

    pattern = "|".join(["quickdraw"])
    df_domain = 100 * df_domain[~df_domain["domain"].str.contains(pattern, case=False, na=False)]

    df_cifar = 100 * df_cifar[df_cifar["severity"].isin([5])]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey="row")

    # --- Bottom row: paired ---
    plot_paired(axes[1, 0], df_id, "IID")
    plot_paired(axes[1, 1], df_cifar, "CIFAR10-C")
    plot_paired(axes[1, 2], df_domain, "DomainNet")

    # --- Top row: CDF ---
    plot_cdf(axes[0, 0], df_id, "IID")
    plot_cdf(axes[0, 1], df_cifar, "CIFAR10-C")
    plot_cdf(axes[0, 2], df_domain, "DomainNet")

    plt.tight_layout()
    plt.tight_layout()
    if save:
        figname = "cdf"
        save_plot(figname=figname)


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire({"gap": plot_gap_distribution_vfm, "cdf": plot_all})


# %% CLI
if __name__ == "__main__":
    main()
# %%
