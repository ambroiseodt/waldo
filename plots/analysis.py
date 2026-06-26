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
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from waldo.infra import FIGURE_DIR, RESULT_DIR

logger = logging.getLogger("waldo.analysis")

# Variables
DATASETS = {
    "aircraft": "Aircraft",
    "cifar10": "Cifar10",
    "cifar10_c-corruption-motion_blur-severity-5": "Motion Blur",
    "cifar100": "Cifar100",
    "dtd": "DTD",
    "domainnet-sketch": "Sketch",
    "eurosat": "EuroSat",
    "flowers102": "Flowers102",
    "food101": "Food-101",
    "pet": "Pet",
}
VFMs = ["aim", "dino", "vit"]
VLMs = ["clip", "siglip"]
LLMs = ["gemma", "llama", "qwen"]
REPRESENTATIVE_MODELS = {"vit": "large", "dinov2": "large", "aimv2": "large", "dinov3": "vitl16"}
REPRESENTATIVE_DATASETS = ["aircraft", "domainnet-sketch", "food101", "flowers102"]
PRETRAINING_TOKENS = {
    "vit": 2.5e11,  # 3.3e11 for ViT-Huge due to patch size 14
    "dinov2": 1.80e12,
    "dinov2_giant": 1.80e12,
    "dinov3": 3.91e12,
    "aimv2": 3.15e12,
}

# Plot aesthetic
WIDTH = 4
HEIGHT = WIDTH
FONTSIZE = 19
LEGFONTSIZE = 10
LINEWIDTH = 3
BARWIDTH = 0.25
MARKER_SIZE = 150
ALPHA_GRID = 0.8


# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
def save_plot(figname: str, dirname: str | None = None, format: str = "pdf", dpi: int = 100) -> None:
    r"""Save figure in pdf format."""
    figure_path = FIGURE_DIR
    if dirname is not None:
        figure_path = figure_path / dirname
    figure_path.mkdir(parents=True, exist_ok=True)
    save_dir = figure_path / f"{figname}.{format}"
    plt.savefig(save_dir, format=format, bbox_inches="tight", dpi=dpi)


# ------------------------------------------------------------------------------
# Vision plotting functions
# ------------------------------------------------------------------------------
def plot_cdf_vision(filename: str, save: bool = False) -> None:
    r"""
    Plot the percentage of runs over VFMs for which the gain is bigger than a certain level.

    Three types of gains over standard probing (over the output of the last transformer block) are considered:
        - gain by optimizing over layers (probing done on the output of transformer blocks)
        - gain by optimizing over modules (probing done over modules of the last transformer block)
        - gain by optimizing over layers and modules (probing done over modules inside transformer blocks)

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Remove VLMs rows from dataframe
    pattern = "|".join(VLMs)
    df = df[~df["model_name"].str.contains(pattern, case=False, na=False)]
    print(f"Results are presented over a total of {len(df)} runs.")

    # Figure
    figsize = 2 * WIDTH, 2 * HEIGHT
    fig = plt.figure(figsize=figsize)

    # Results
    for col, label in [("gain_layer", "Layer"), ("gain_module", "Module"), ("gain", "Location")]:
        gains = np.sort(df[col].to_numpy())
        percentage = 100 * (1 - np.arange(1, len(gains) + 1) / len(gains))
        plt.plot(gains, percentage, linewidth=LINEWIDTH, label=label)

    # Visualization
    ax = fig.axes[0]
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xlabel("Gain over standard probing", fontsize=FONTSIZE)
    ax.set_ylabel(r"Percentage of runs with gain $\geq x$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"cdf_vision_{filename}"
        save_plot(figname=figname)


def plot_dataset_difficulty_vision(filename: str, save: bool = False) -> None:
    r"""
    Plot the evolution of gain over standard probing, averaged over models for a given dataset,
    as a function of the dataset difficulty (measured by the highest achievable performance).

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Remove VLMs rows from dataframe
    pattern = "|".join(VLMs)
    df = df[~df["model_name"].str.contains(pattern, case=False, na=False)]
    print(f"Results are presented over a total of {len(df)} runs.")

    # Figure
    figsize = 2 * WIDTH, 2 * HEIGHT
    fig = plt.figure(figsize=figsize)

    # Results
    summary = (
        df.groupby("dataset_name")
        .agg(
            max_accuracy=("baseline_accuracy", "max"),
            mean_gain=("gain", "mean"),
        )
        .reset_index()
    )
    rho, pval = spearmanr(summary["max_accuracy"], summary["mean_gain"])
    sns.regplot(data=summary, x="max_accuracy", y="mean_gain", scatter=False, ci=None, line_kws={"linewidth": 2})
    plt.scatter(summary["max_accuracy"], summary["mean_gain"], s=60, zorder=3)

    for _, row in summary.iterrows():
        plt.annotate(
            DATASETS[row["dataset_name"]],
            (row["max_accuracy"], row["mean_gain"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=LEGFONTSIZE,
        )

    # Visualization
    ax = fig.axes[0]
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xlabel("Maximum achievable accuracy", fontsize=FONTSIZE)
    ax.set_ylabel(r"Average gain from location optimization", fontsize=FONTSIZE)
    ax.set_title(f"Dataset difficulty vs gain\nSpearman ρ = {rho:.2f}, p = {pval:.2e}", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"dataset_difficulty_vision_{filename}"
        save_plot(figname=figname)


def plot_model_family_vision(filename: str, save: bool = False, ncol: int = 3) -> None:
    r"""
    Plot the gain over standard probing for each model family.

    Three types of gains over standard probing (over the output of the last transformer block) are considered:
        - gain by optimizing over layers (probing done on the output of transformer blocks)
        - gain by optimizing over modules (probing done over modules of the last transformer block)
        - gain by optimizing over layers and modules (probing done over modules inside transformer blocks)

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    ncol: int, default=3
        Number of columns in the legend.
    """
    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Remove VLMs rows from dataframe
    pattern = "|".join(VLMs)
    df = df[~df["model_name"].str.contains(pattern, case=False, na=False)]

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_MODELS.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = df[mask]

    # Figure
    N = len(REPRESENTATIVE_DATASETS)
    figsize = 2 * N * WIDTH, 2 * HEIGHT
    fig, axes = plt.subplots(1, N, figsize=figsize, sharey=True)
    model_order = list(REPRESENTATIVE_MODELS.keys())
    gain_types = ["gain_layer", "gain_module", "gain"]
    gain_labels = ["Layer", "Module", "Both"]

    colors = sns.color_palette("Set2", 3)

    # Results
    for ax, dataset in zip(axes, REPRESENTATIVE_DATASETS, strict=False):
        sub = df[df["dataset_name"] == dataset]
        agg = sub.groupby("model_name")[gain_types].mean().reindex(model_order)
        x = np.arange(len(model_order))
        for i, (g, label, color) in enumerate(zip(gain_types, gain_labels, colors, strict=False)):
            ax.bar(x + i * BARWIDTH, agg[g].values, width=BARWIDTH, label=label, color=color)

        # Visualization
        ax.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1, labelsize=FONTSIZE)
        ax.set_title(dataset, fontsize=FONTSIZE)
        ax.set_xticks(x + BARWIDTH)
        ax.set_xticklabels(model_order, rotation=20)
        if ax == axes[0]:
            ax.set_ylabel("Gain over standard probing", fontsize=FONTSIZE)
        sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=True,
        handlelength=1.9,
        fontsize=FONTSIZE,
    )

    plt.tight_layout()
    if save:
        figname = f"model_family_vision_{filename}"
        save_plot(figname=figname)


def plot_pretraining_scale_vision(filename: str, save: bool = False) -> None:
    r"""
    Plot the evolution of gain over standard probing as a function of pretraining scale,
    measured by the number of images seen during pretraining.

    We follow the scaling law literature for LLMs [1, 2]_ and consider the total amount of tokens seen during
    pretraining. Notably, this takes into considerations multiple epochs, i.e., the number of image exposure, and
    the number of tokens in each image. This depends on the patch size and image resolution. For simplicity (and
    since this does not impact the order of magnitude), we only consider visual tokens. In particular, we do not
    account for CLS or register tokens. In summary, we have: image exposure x token per image.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.

    Notes
    -----
    We recall that for an RGB image H x W x C, and a patch size P, the nb of tokens in the sequence is N = (HW / P^2).
    For each model, we assume the following amount of pretraining images (using the original technical reports):
        - ViT [3]_: trained on ImageNet-21K with 14M images
        - DINOV2 [4]_: trained on LVD-142M with 142B images
        - DINOV3 [5]_: trained on LVD-1689M with 1.7B images
        - AIMv2 [6]_: trained on a mixture of image-text pairs with 12B distinct pairs
    Some models are trained from scracth, while others are obtained via knowledge distillation. For simplicity, we
    consider that distilled models have seen the same amount of pretraining images as their teacher model. For models
    trained using epochs, we simply multiply the dataset size by the number of epochs to gain the number of exposures.
    For models trained by iterations, we multiply the total number of iterations by the batch size.


    References
    ----------
    .. [1] J. Kaplan et al. Scaling Laws for Neural Language Models. arXiv, 2020

    .. [2] J. Hoffman et al. Training Compute-Optimal Large Language Models. In NeurIPS 2022

    .. [3] A. Dosovitskiy, L. Beyer et al. An Image is Worth 16x16 Words: Transformers
       for Image Recognition at Scale. In ICLR 2021

    .. [4] M. Oquab et al. DINOv2: Learning Robust Visual Feature without Supervision. In TMLR 2024

    .. [5] O. Simeoni et al. DINOv3. arXiv 2025

    .. [6] E. Fini et al. Multimodal Autoregressive Pre-training of Large Vision Encoders. In CVPR 2025
    """

    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_MODELS.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = df[mask]

    # map pretraining scale
    df["pretraining_images"] = df["model_name"].map(PRETRAINING_TOKENS)

    plt.figure(figsize=(7, 4))

    # color by dataset (cleanest visual encoding)
    datasets = df["dataset_name"].unique()  # REPRESENTATIVE_DATASETS
    palette = sns.color_palette("tab10", len(datasets))
    color_map = dict(zip(datasets, palette))
    for ds in datasets:
        sub = df[df["dataset_name"] == ds]

        plt.scatter(
            sub["pretraining_images"],
            sub["gain"],
            label=ds,
            color=color_map[ds],
            s=80,
            alpha=0.9,
        )

        # optional: connect points within dataset (ONLY visual grouping)
        sub = sub.sort_values("pretraining_images")
        if len(sub) > 1:
            plt.plot(
                sub["pretraining_images"],
                sub["gain"],
                color=color_map[ds],
                alpha=0.5,
            )

    plt.xscale("log")
    plt.xlabel("Pretraining scale (number of images)")
    plt.ylabel("Location gain")

    plt.grid(alpha=0.3)

    # Common legend
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"pretraining_scale_vision_{filename}"
        save_plot(figname=figname)


def plot_model_size_vision(filename: str, save: bool = False, ncol: int = 1) -> None:
    r"""
    TODO

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    ncol: int, default=1
        Number of columns in the legend.
    """
    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Remove VLMs rows from dataframe
    pattern = "|".join(VLMs)
    df = df[~df["model_name"].str.contains(pattern, case=False, na=False)]

    # Recover representative datasets and models
    datasets = df["dataset_name"].unique()  # ["aircraft", "domainnet-sketch", "food101", "flowers102"]
    models = ["vit", "dinov2", "dinov3", "aimv2"]
    pattern = "|".join(models)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Figures
    N = len(models)
    figsize = (2 * N * WIDTH, 2 * HEIGHT)
    fig, axes = plt.subplots(1, N, figsize=figsize, sharey=True)
    palette = sns.color_palette("Set2", len(datasets))

    # # ------------------------------------------------------------
    # # One subplot per family
    # # ------------------------------------------------------------
    # for ax, family in zip(axes, models):
    #     sub = df[df["model_name"] == family]

    #     # preserve original order of sizes as they appear
    #     size_order = sub[["model_size"]].drop_duplicates()["model_size"].tolist()

    #     for dataset, color in zip(datasets, palette):
    #         ds = sub[sub["dataset_name"] == dataset]

    #         summary = ds.groupby("model_size")["gain"].mean().reindex(size_order)

    #         ax.plot(
    #             np.arange(len(summary)),
    #             summary.values,
    #             marker="o",
    #             linewidth=2,
    #             label=dataset,
    #             color=color,
    #         )

    #     ax.set_title(family, fontsize=14)

    #     ax.set_xticks(np.arange(len(size_order)))
    #     ax.set_xticklabels(
    #         size_order,
    #         rotation=20,
    #         fontsize=11,
    #     )

    #     ax.tick_params(axis="y", labelsize=11)

    #     ax.set_xlabel("Model size", fontsize=12)

    # ------------------------------------------------------------
    # Plot per family
    # ------------------------------------------------------------
    for ax, family in zip(axes, models, strict=False):
        sub = df[df["model_name"] == family]

        # Ensure consistent ordering of sizes
        size_order = sorted(sub["model_size"].unique())

        sns.boxplot(
            data=sub, x="model_size", y="gain", order=size_order, ax=ax, color="lightsteelblue", showfliers=False
        )

        # optional: overlay mean trend (very helpful visually)
        medians = sub.groupby("model_size")["gain"].median().reindex(size_order)

        ax.plot(
            range(len(size_order)),
            medians.values,
            color="black",
            marker="o",
            linewidth=2,
            label="mean",
        )

        ax.set_title(family, fontsize=14)
        ax.set_xlabel("Model size", fontsize=12)
        ax.tick_params(axis="x", labelrotation=20, labelsize=11)
        ax.tick_params(axis="y", labelsize=11)

    axes[0].set_ylabel(
        "Best gain (optimized layer + module)",
        fontsize=12,
    )

    axes[0].legend(
        title="Dataset",
        fontsize=10,
        title_fontsize=11,
    )

    plt.suptitle(
        "Impact of model scale on recoverable probing gain",
        fontsize=16,
        y=1.03,
    )
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"model_size_vision_{filename}"
        save_plot(figname=figname)


def test_size_effect(filename: str):
    """
    Mixed-effects model:
    gain ~ model_size + random effects (dataset, family)

    Returns model summary and p-value for size effect.
    """

    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Remove VLMs rows from dataframe
    pattern = "|".join(VLMs)
    df = df[~df["model_name"].str.contains(pattern, case=False, na=False)]

    # # ensure categorical variables are treated correctly
    # df["dataset_name"] = df["dataset_name"].astype("category")
    # df["model_name"] = df["model_name"].astype("category")

    # # IMPORTANT:
    # # model_size should be numeric (e.g., log params is even better)
    # # if not numeric, convert it first
    # # df["model_size"] = df["model_size"].astype(float)

    # model = smf.mixedlm(
    #     "gain ~ model_size",
    #     df,
    #     groups=df["model_name"],  # family-level random effect
    #     re_formula="~1",
    # )

    # result = model.fit()

    # print(result.summary())

    # print("\nP-value for size effect:")
    # print(result.pvalues.get("model_size", None))

    # ensure categorical structure
    df["model_name"] = df["model_name"].astype("category")
    df["dataset_name"] = df["dataset_name"].astype("category")

    # # ------------------------------------------------------------
    # # mixed model
    # # ------------------------------------------------------------
    # model = smf.mixedlm(
    #     formula="gain ~ model_size",
    #     data=df,
    #     groups=df["model_name"],  # model family random effect
    #     vc_formula={"dataset": "0 + C(dataset_name)"},  # dataset random effect
    # )

    # result = model.fit()

    # print(result.summary())

    # # ------------------------------------------------------------
    # # interpretability helper
    # # ------------------------------------------------------------
    # print("\n=== Fixed effect (model_size) ===")
    # for k, v in result.pvalues.items():
    #     if "model_size" in k:
    #         print(f"{k}: p = {v:.4g}")

    # ------------------------------------------------------------
    # 2. MIXED EFFECT MODEL (STATISTICAL TEST)
    # ------------------------------------------------------------
    model = smf.mixedlm("gain ~ model_size", df, groups=df["model_name"], vc_formula={"dataset": "0 + C(dataset_name)"})

    result = model.fit()

    print("\n===== MIXED EFFECT MODEL =====")
    print(result.summary())

    # ------------------------------------------------------------
    # 3. VARIANCE BREAKDOWN (VERY IMPORTANT FOR PAPER)
    # ------------------------------------------------------------
    print("\n===== VARIANCE COMPONENTS =====")
    print("\n===== VARIANCE COMPONENTS =====")

    # dataset variance (vc component)
    dataset_var = result.vcomp[0] if hasattr(result, "vcomp") and len(result.vcomp) > 0 else 0.0

    # family variance (random effect)
    if hasattr(result, "cov_re") and result.cov_re is not None and result.cov_re.size > 0:
        family_var = float(np.asarray(result.cov_re).flatten()[0])
    else:
        family_var = 0.0

    # residual variance
    residual_var = float(result.scale)

    total = dataset_var + family_var + residual_var

    def safe_pct(x):
        return (x / total) if total > 0 else 0

    print(f"Dataset variance: {dataset_var:.3f} ({safe_pct(dataset_var):.2%})")
    print(f"Family variance:  {family_var:.3f} ({safe_pct(family_var):.2%})")
    print(f"Residual:        {residual_var:.3f} ({safe_pct(residual_var):.2%})")

    return result


def get_table_modules() -> None:
    r"""TODO: we want to show that act is overall the best module (check frequency of best module, or percentage
    of optimal performance recovered compared to other modules and to standard probing). So that both
    when act is optimized over layers and when act is at the last layer.
    """
    pass


def get_table_ranking() -> None:
    r"""Showcase how the ranking among models (best among family and between families) would change if the best
    probing was considerd and the same if act probing was chosen.
    """
    pass


# ------------------------------------------------------------------------------
# Multimodal plotting functions
# ------------------------------------------------------------------------------
def plot_cdf_multimodal(filename: str, save: bool = False) -> None:
    r"""
    Plot the percentage of runs over VLMs for which the gain is bigger than a certain level.

    Three types of gains over standard probing (over the output of the last transformer block) are considered:
        - gain by optimizing over layers (probing done on the output of transformer blocks)
        - gain by optimizing over modules (probing done over modules of the last transformer block)
        - gain by optimizing over layers and modules (probing done over modules inside transformer blocks)

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")

    # Keep only VLMs rows from dataframe
    pattern = "|".join(VLMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    print(f"Results are presented over a total of {len(df)} runs.")

    # Figure
    figsize = 2 * WIDTH, 2 * HEIGHT
    fig = plt.figure(figsize=figsize)
    for col, label in [("gain_layer", "Layer"), ("gain_module", "Module"), ("gain", "Location")]:
        gains = np.sort(df[col].to_numpy())
        percentage = 100 * (1 - np.arange(1, len(gains) + 1) / len(gains))
        plt.plot(gains, percentage, linewidth=LINEWIDTH, label=label)

    # Visualization
    ax = fig.axes[0]
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xlabel("Gain over standard probing", fontsize=FONTSIZE)
    ax.set_ylabel(r"Percentage of runs with gain $\geq x$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"cdf_multimodal_{filename}"
        save_plot(figname=figname)


# ------------------------------------------------------------------------------
# Language plotting functions
# ------------------------------------------------------------------------------
def plot_cdf_language(filename: str, save: bool = False) -> None:
    r"""
    Plot the percentage of runs over LLMs for which the gain is bigger than a certain level.

    Three types of gains over standard probing (over the output of the last transformer block) are considered:
        - gain by optimizing over layers (probing done on the output of transformer blocks)
        - gain by optimizing over modules (probing done over modules of the last transformer block)
        - gain by optimizing over layers and modules (probing done over modules inside transformer blocks)

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    print(f"Results are presented over a total of {len(df)} runs.")

    # Figure
    figsize = 2 * WIDTH, 2 * HEIGHT
    fig = plt.figure(figsize=figsize)
    for col, label in [("gain_layer", "Layer"), ("gain_module", "Module"), ("gain", "Location")]:
        gains = np.sort(df[col].to_numpy())
        percentage = 100 * (1 - np.arange(1, len(gains) + 1) / len(gains))
        plt.plot(gains, percentage, linewidth=LINEWIDTH, label=label)

    # Visualization
    ax = fig.axes[0]
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xlabel("Gain over standard probing", fontsize=FONTSIZE)
    ax.set_ylabel(r"Percentage of runs with gain $\geq x$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"cdf_language_{filename}"
        save_plot(figname=figname)


# def plot_gap_distribution_vfm(save: bool = False) -> None:
#     # Recover results
#     path = RESULT_DIR / "analysis" / "cls.csv"
#     df = pd.read_csv(path)
#     pattern = "|".join(VLMs)
#     df = df[~df["model_name"].str.contains(pattern, case=False, na=False)]
#     # pattern = "|".join(["dinov3"])
#     # df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
#     # pattern = "|".join(["vits"])
#     # df = df[df["model_size"].str.contains(pattern, case=False, na=False)]
#     # pattern = "|".join(["1B", "3B", "vit7b16", "giant", "huge", "large"])
#     # df = df[~df["model_size"].str.contains(pattern, case=False, na=False)]
#     # df = df[df["severity"].isin([5])]
#     # pattern = "|".join(["quickdraw"])
#     # df = df[~df["domain"].str.contains(pattern, case=False, na=False)]
#     plot_df = pd.DataFrame(
#         {
#             "gain": pd.concat(
#                 [
#                     df["gain_layer"],
#                     df["gain_module"],
#                     df["gain"],
#                 ],
#                 ignore_index=True,
#             ),
#             "type": (["best layer"] * len(df) + ["best module"] * len(df) + ["best location"] * len(df)),
#             # "type": (["best layer"] * len(df) + ["best module"] * len(df)),
#         }
#     )
#     plot_df["gain"] = 100 * plot_df["gain"]
#     plt.figure(figsize=(6, 5))

#     sns.boxplot(
#         data=plot_df,
#         x="type",
#         y="gain",
#         showfliers=False,  # IMPORTANT: removes outlier clutter
#         showmeans=True,
#     )

#     # sns.stripplot(data=plot_df, x="type", y="gap", color="black", alpha=0.25, size=2)

#     plt.axhline(0, color="gray", linestyle="--", linewidth=1)

#     plt.xlabel("")
#     plt.ylabel("Gain over standard probing")
#     plt.title("Probe-location optimization improves performance")

#     plt.tight_layout()
#     if save:
#         figname = "gap_distribution"
#         save_plot(figname=figname)


# # def plot_cdf(ax, df, title):
# #     for col, label in [("gain_layer", "Layer"), ("gain_module", "Module"), ("gain", "Location")]:
# #         x = np.sort(df[col].values)
# #         y = np.arange(1, len(x) + 1) / len(x)
# #         ax.plot(x, 1 - y, linewidth=2, label=label)

# #     ax.set_title(title)
# #     ax.set_xlabel("Gain")
# #     ax.set_ylabel("CDF")
# #     ax.grid(alpha=0.3)
# #     ax.legend()


# def plot_paired(ax, df, title):
#     data = pd.DataFrame({"Layer": df["gain_layer"], "Module": df["gain_module"], "Location": df["gain"]})

#     melted = data.melt(var_name="Method", value_name="Gain")

#     sns.barplot(
#         data=melted,
#         x="Method",
#         y="Gain",
#         ax=ax,
#         errorbar=("ci", 95),  # <-- THIS adds std bars
#     )

#     ax.axhline(0, linestyle="--", color="black", linewidth=1)
#     ax.set_title(title)
#     ax.set_ylabel("Gain (mean ± std)")
#     ax.grid(axis="y", alpha=0.3)


# def plot_all(save: bool = False):

#     path = RESULT_DIR / "analysis"
#     df_id = pd.read_csv(path / "cls.csv")
#     df_domain = pd.read_csv(path / "domainnet.csv")
#     df_cifar = pd.read_csv(path / "cifar10_c.csv")
#     pattern = "|".join(VLMs)
#     df_id = 100 * df_id[~df_id["model_name"].str.contains(pattern, case=False, na=False)]
#     df_domain = df_domain[~df_domain["model_name"].str.contains(pattern, case=False, na=False)]
#     df_cifar = df_cifar[~df_cifar["model_name"].str.contains(pattern, case=False, na=False)]

#     pattern = "|".join(["quickdraw"])
#     df_domain = 100 * df_domain[~df_domain["domain"].str.contains(pattern, case=False, na=False)]

#     df_cifar = 100 * df_cifar[df_cifar["severity"].isin([5])]

#     fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey="row")

#     # --- Bottom row: paired ---
#     plot_paired(axes[1, 0], df_id, "IID")
#     plot_paired(axes[1, 1], df_cifar, "CIFAR10-C")
#     plot_paired(axes[1, 2], df_domain, "DomainNet")

#     # --- Top row: CDF ---
#     plot_cdf(axes[0, 0], df_id, "IID")
#     plot_cdf(axes[0, 1], df_cifar, "CIFAR10-C")
#     plot_cdf(axes[0, 2], df_domain, "DomainNet")

#     plt.tight_layout()
#     plt.tight_layout()
#     if save:
#         figname = "cdf"
#         save_plot(figname=figname)


# def summarize_models(df):

#     summary = (
#         df.groupby(["model_name", "model_size"])
#         .agg(
#             gain_mean=("gain", "mean"),
#             gain_std=("gain", "std"),
#             gain_layer_mean=("gain_layer", "mean"),
#             gain_layer_std=("gain_layer", "std"),
#             gain_module_mean=("gain_module", "mean"),
#             gain_module_std=("gain_module", "std"),
#         )
#         .reset_index()
#     )

#     return summary.sort_values(["model_name", "model_size"])


def dataset_level_gain(df):

    summary = (
        df.groupby("dataset_name")
        .agg(
            acc_mean=("baseline_accuracy", "mean"),
            acc_std=("baseline_accuracy", "std"),
            acc_best=("baseline_accuracy", "max"),
            gain_mean=("gain", "mean"),
            gain_std=("gain", "std"),
            gain_layer_mean=("gain_layer", "mean"),
            gain_layer_std=("gain_layer", "std"),
            gain_module_mean=("gain_module", "mean"),
            gain_module_std=("gain_module", "std"),
        )
        .reset_index()
    )

    return summary.sort_values("gain_mean", ascending=False)


# def corruption_level_gain(df):

#     summary = (
#         df.groupby(["corruption", "severity"])
#         .agg(
#             acc_mean=("baseline_accuracy", "mean"),
#             acc_std=("baseline_accuracy", "std"),
#             acc_best=("baseline_accuracy", "max"),
#             gain_mean=("gain", "mean"),
#             gain_std=("gain", "std"),
#             gain_layer_mean=("gain_layer", "mean"),
#             gain_layer_std=("gain_layer", "std"),
#             gain_module_mean=("gain_module", "mean"),
#             gain_module_std=("gain_module", "std"),
#         )
#         .reset_index()
#     )

#     return summary.sort_values("gain_mean", ascending=False)


# def domain_level_gain(df):

#     summary = (
#         df.groupby("domain")
#         .agg(
#             acc_mean=("baseline_accuracy", "mean"),
#             acc_std=("baseline_accuracy", "std"),
#             acc_best=("baseline_accuracy", "max"),
#             gain_mean=("gain", "mean"),
#             gain_std=("gain", "std"),
#             gain_layer_mean=("gain_layer", "mean"),
#             gain_layer_std=("gain_layer", "std"),
#             gain_module_mean=("gain_module", "mean"),
#             gain_module_std=("gain_module", "std"),
#         )
#         .reset_index()
#     )

#     return summary.sort_values("gain_mean", ascending=False)


# def task_level_gain(df):

#     summary = (
#         df.groupby("task_name")
#         .agg(
#             acc_mean=("baseline_accuracy", "mean"),
#             acc_std=("baseline_accuracy", "std"),
#             acc_best=("baseline_accuracy", "max"),
#             gain_mean=("gain", "mean"),
#             gain_std=("gain", "std"),
#             gain_layer_mean=("gain_layer", "mean"),
#             gain_layer_std=("gain_layer", "std"),
#             gain_module_mean=("gain_module", "mean"),
#             gain_module_std=("gain_module", "std"),
#         )
#         .reset_index()
#     )

#     return summary.sort_values("gain_mean", ascending=False)


# def summarize_llms(df):

#     summary = (
#         df.groupby(["model_name"])
#         .agg(
#             gain_mean=("gain", "mean"),
#             gain_std=("gain", "std"),
#             gain_layer_mean=("gain_layer", "mean"),
#             gain_layer_std=("gain_layer", "std"),
#             gain_module_mean=("gain_module", "mean"),
#             gain_module_std=("gain_module", "std"),
#         )
#         .reset_index()
#     )

#     return summary.sort_values(["model_name"])


def table():
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / "cls.csv")
    pattern = "|".join(VLMs)
    df = df[~df["model_name"].str.contains(pattern, case=False, na=False)]
    print(dataset_level_gain(df))


# ------------------------------------------------------------------------------
# Results functions
# ------------------------------------------------------------------------------
def plot_figures() -> None:
    # # CDF
    # save = True
    # for filename in ["cls", "cifar10_c", "domainnet"]:
    #     plot_cdf_vision(filename=filename, save=save)
    #     plot_cdf_multimodal(filename=filename, save=save)
    # plot_cdf_language(filename="mteb", save=save)

    # # Dataset difficulty
    # save = True
    # filename = "cls"
    # plot_dataset_difficulty_vision(filename=filename, save=save)

    # Model family
    save = True
    filename = "cls"
    plot_model_family_vision(filename=filename, save=save)

    # Pretraining scale
    save = True
    filename = "cls"
    plot_pretraining_scale_vision(filename=filename, save=save)

    # # Model size
    # save = True
    # filename = "cls"
    # # plot_model_size_vision(filename=filename, save=save)
    # print(test_size_effect(filename=filename))


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire({"plot": plot_figures, "table": table})


# %% CLI
if __name__ == "__main__":
    main()
# %%
