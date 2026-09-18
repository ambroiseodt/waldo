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
from matplotlib.container import ErrorbarContainer
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare, spearmanr

from waldo.infra import FIGURE_DIR, RESULT_DIR

logger = logging.getLogger("waldo.analysis")

# Variables
MODULE_ORDER = ["ln1", "mha", "rc1", "ln2", "fc1", "act", "fc2", "rc2"]
LABEL = {
    "oracle": r"$\mathscr{G}_\mathrm{joint}$",
    "layer": r"$\mathscr{G}_\mathrm{layer}$",
    "module": r"$\mathscr{G}_\mathrm{module}$",
}

# VFMs
DATASETS = {
    "aircraft": "Aircraft",
    "cifar10": "Cifar10",
    "cifar10_c-corruption-contrast-severity-5": "Contrast",
    "cifar10_c-corruption-motion_blur-severity-5": "Motion Blur",
    "cifar10_c-corruption-speckle_noise-severity-5": "Speckle Noise",
    "cifar100": "Cifar100",
    "dtd": "DTD",
    "domainnet-clipart": "Clipart",
    "domainnet-infograph": "Infograph",
    "domainnet-quickdraw": "Quickdraw",
    "domainnet-painting": "Painting",
    "domainnet-sketch": "Sketch",
    "eurosat": "EuroSat",
    "flowers102": "Flowers102",
    "food101": "Food-101",
    "iwildcam": "iWildCam",
    "pet": "Pet",
    "rxrx1": "RxRx1",
}
VFMs = ["aim", "dino", "vit"]
REPRESENTATIVE_VFMs = {"vit": "large", "dinov2": "large", "dinov3": "vitl16", "aimv2": "large"}
VFM_NAMES = {"vit": "ViT", "dinov2": "DINOv2", "dinov3": "DINOv3", "aimv2": "AIMv2"}
VFM_SIZES = {
    "vit": ["base", "large", "huge"],
    "dinov2": ["small", "base", "large", "giant"],
    "aimv2": ["large", "huge", "1B", "3B"],
    "dinov3": ["vits16", "vitb16", "vitl16", "vit7b16"],
}
REPRESENTATIVE_DATASETS = ["cifar10", "domainnet-sketch", "domainnet-quickdraw"]
VFM_MARKERS = {
    "vit": "o",
    "dinov2": "^",
    "aimv2": "*",
    "dinov3": "v",
}
VFM_MARKER_SIZE = {
    "vit": 150,
    "dinov2": 150,
    "aimv2": 220,
    "dinov3": 150,
}

# LLMs
LLMs = ["Qwen3", "Llama-3", "gemma-3"]
REPRESENTATIVE_LLMs = {"qwen3": "Qwen3-1.7B", "llama3": "Llama-3.2-1B", "gemma3": "gemma-3-1b-pt"}
LLM_NAMES = {"qwen3": "Qwen3", "llama3": "Llama3", "gemma3": "Gemma3"}
LLM_FAMILIES = {
    "qwen3": ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"],
    "llama3": ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B"],
    "gemma3": ["gemma-3-1b-pt", "gemma-3-4b-pt", "gemma-3-12b-pt"],
}
LLM_SIZES = {
    "Qwen3-1.7B": 1.7,
    "Qwen3-4B": 4,
    "Qwen3-8B": 8,
    "Llama-3.2-1B": 1,
    "Llama-3.2-3B": 3,
    "Llama-3.1-8B": 8,
    "gemma-3-1b-pt": 1,
    "gemma-3-4b-pt": 4,
    "gemma-3-12b-pt": 12,
}

# Plot aesthetic
FONTSIZE = 21
LEG_FONTSIZE = FONTSIZE
LINEWIDTH = 4
BARWIDTH = 0.25
MARKER_LINEWIDTH = 0.5
ALPHA_GRID = 0.8
ZOOM_SCALE = 0.7
GAIN_PALETTE = {"layer": "#7987b8", "module": "#9fc9b1", "oracle": "#c6a4e4"}
MODULE_PALETTE = {
    "ln1": "#daa4ac",
    "mha": "#37abb5",
    "rc1": "#d0e2c2",
    "ln2": "#b153a1",
    "fc1": "#a291e1",
    "act": "#96c0cf",
    "fc2": "#858ec2",
    "rc2": "#428379",
}
LLM_PALETTE = {"qwen3": "#7987b8", "llama3": "#9fc9b1", "gemma3": "#c6a4e4"}
CMAP = "viridis"


# Visual parameters
palette = sns.cubehelix_palette()
custom_params = {"axes.grid": False}
sns.set_theme(style="ticks", palette=palette, rc=custom_params)
sns.set_context("talk")
plt.rcParams.update({"figure.autolayout": True})
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"


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
# Runs statistics
# ------------------------------------------------------------------------------
def table_runs() -> None:
    r"""Compute the number of runs in our experiments."""
    path = RESULT_DIR

    # Vision
    comp = 0
    for filename in ["cls", "cifar10_c"]:
        df = pd.read_csv(path / f"{filename}.csv")
        df = df.copy()
        pattern = "|".join(VFMs)
        df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
        print(f"There are {len(df)} runs for {filename}.")
        comp += len(df)
    print(f"There are {comp} VFMs runs.")

    # Language
    filename = "mteb"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()
    pattern = "|".join(LLMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    comp += len(df)
    print(f"There are {len(df)} LLMs runs.")

    # Total
    print(f"There are a total of {comp} runs.")


# ------------------------------------------------------------------------------
# VFMs plotting functions
# ------------------------------------------------------------------------------
def plot_cdf_vision(filename: str, ncol: int = 1, save: bool = False) -> None:
    r"""
    Plot the percentage of runs over VFMs for which the gain is strictly above a certain threshold.

    Three types of gains over standard probing (over the output of the last transformer block) are considered:
        - gain by optimizing over layers (probing done on the output of transformer blocks)
        - gain by optimizing over modules (probing done over modules of the last transformer block)
        - gain by optimizing over layers and modules (probing done over modules inside transformer blocks)

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    ncol: int, default=1
        Number of legend columns.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 5
    height = 4
    figsize = width, height
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = df[df["dataset_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    for col, label in [("gain_layer", "layer"), ("gain_module", "module"), ("gain", "oracle")]:
        gains = np.sort(df[col].to_numpy())
        percentage = 100 * (1 - np.arange(1, len(gains) + 1) / len(gains))
        ax.plot(gains, percentage, linewidth=LINEWIDTH, color=GAIN_PALETTE[label], label=LABEL[label])

    # Visualization (parent)
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xlim(-0.1, 10)
    ax.set_ylim(-1, 101)
    yticks = [0, 25, 50, 75, 100]
    ax.set_yticks(yticks)
    ax.set_xlabel("Gain over standard probing (pp)", fontsize=FONTSIZE)
    label = ax.set_ylabel(r"Probing runs with gain $> x$", fontsize=FONTSIZE)
    x, y = label.get_position()
    label.set_position((x, 0.4))
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    lines, labels = lines[::-1], labels[::-1]
    legend = fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.68, 0.9),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=True,
        fontsize=LEG_FONTSIZE,
    )
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(10)
    plt.tight_layout()
    if save:
        figname = f"cdf_vision_{filename}"
        save_plot(figname=figname)


def plot_transfer_difficulty_vision(filename: str, save: bool = False) -> None:
    r"""
    Plot dataset-level mean oracle gain as a function of empirical transfer difficulty.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 5.5
    height = 5
    figsize = width, height
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = df[df["dataset_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    summary = (
        df.groupby("dataset_name")
        .agg(
            mean_standard_accuracy=("baseline_accuracy", "mean"),
            mean_gain=("gain", "mean"),
            std_gain=("gain", "std"),
            n_models=("gain", "count"),
        )
        .reset_index()
    )
    summary["transfer_error"] = 100 - summary["mean_standard_accuracy"]

    # 95% confidence interval (nonnegative lower bound since oracle gain values are nonnegative)
    summary["margin"] = 1.96 * summary["std_gain"] / np.sqrt(summary["n_models"])
    summary["ci_upper"] = summary["mean_gain"] + summary["margin"]
    summary["ci_lower"] = summary["mean_gain"] - summary["margin"]
    summary["ci_lower"] = summary["ci_lower"].clip(lower=0)

    # Plot
    ax.scatter(
        summary["transfer_error"],
        summary["mean_gain"],
        s=160,
        color=GAIN_PALETTE["oracle"],
        edgecolors="#333333",
        linewidths=0.5,
        zorder=3,
    )
    ax.errorbar(
        summary["transfer_error"],
        summary["mean_gain"],
        yerr=[
            summary["mean_gain"] - summary["ci_lower"],
            summary["ci_upper"] - summary["mean_gain"],
        ],
        fmt="none",
        ecolor="#333333",
        elinewidth=0.5,
        capsize=3,
        capthick=0.5,
        zorder=2,
    )
    for container in ax.containers:
        if not isinstance(container, ErrorbarContainer):
            continue

        _, caplines, barlinecols = container.lines

        for capline in caplines:
            capline.set_color("#333333")
            capline.set_linewidth(0.5)
            capline.set_alpha(0.55)

        for barlinecol in barlinecols:
            barlinecol.set_color("#333333")
            barlinecol.set_linewidth(0.5)
            barlinecol.set_linestyle("--")
            barlinecol.set_alpha(0.55)

    # Spearman correlation rank test
    rho, pval = spearmanr(summary["transfer_error"], summary["mean_gain"])
    print(f"Spearman coefficient is {rho:2f} with p-value {pval:2f}.")

    # Linear regression for visual trend
    sns.regplot(
        data=summary,
        x="transfer_error",
        y="mean_gain",
        color=GAIN_PALETTE["oracle"],
        scatter=False,
        ci=None,
        line_kws={"linewidth": 2, "alpha": 0.8},
        ax=ax,
    )

    # Annotate
    summary_subset = summary.copy()
    summary_subset = summary_subset[summary_subset["dataset_name"].isin(REPRESENTATIVE_DATASETS)]
    for _, row in summary_subset.iterrows():
        ax.annotate(
            DATASETS[row["dataset_name"]],
            (row["transfer_error"], row["mean_gain"]),
            xytext=(8, -1.5),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=18,
            color="#333333",
            zorder=10,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.0),
        )

    # Spearman correlation legend
    handle = Line2D(
        [0],
        [0],
        color=GAIN_PALETTE["oracle"],
        linewidth=2,
        label=rf"$\rho= {rho:.2f}$"
        + "\n"
        + rf"$\text{{p-value}}= {pval:.1e}$".replace("e-0", "e-").replace("e+0", "e+"),
    )
    legend = ax.legend(
        handles=[handle],
        loc="upper center",
        bbox_to_anchor=(0.7, 0.24),
        fancybox=True,
        borderaxespad=0,
        shadow=False,
        frameon=True,
        handlelength=1.9,
        fontsize=15,
    )
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1)
    legend.set_zorder(10)

    # Visualization
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, 7.1)
    xticks = [0, 25, 50, 75, 100]
    ax.set_xticks(xticks)
    yticks = [0, 3.5, 7.0]
    ax.set_yticks(yticks)
    ax.set_xlabel(r"Mean probing error ($\%$)", fontsize=FONTSIZE)
    ax.set_ylabel(rf"Oracle gain {LABEL['oracle']} (pp)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        figname = f"transfer_difficulty_vision_{filename}"
        save_plot(figname=figname)


def plot_cifar10c_severity(filename: str, save: bool = False) -> None:
    r"""
    Plot probing gain as a function of corruption severity on CIFAR10-C.

    Each curve corresponds to one representative vision model.
    Results are averaged over corruption types.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 5.5
    height = 5
    figsize = width, height
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Corruption severity
    severity_order = [1, 3, 5]
    sns.barplot(
        data=df,
        x="severity",
        y="gain",
        order=severity_order,
        estimator=np.mean,
        errorbar=("ci", 95),
        color=GAIN_PALETTE["oracle"],
        legend=False,
        err_kws={
            "color": "#333333",
            "linewidth": 0.5,
            "linestyle": "--",
        },
        capsize=0.15,
        ax=ax,
    )

    # Visualization
    ax.grid(axis="y", alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    yticks = [0.0, 1.2, 2.4]
    ax.set_yticks(yticks)
    ax.set_xlabel("Corruption severity", fontsize=FONTSIZE)
    ax.set_ylabel(rf"Oracle gain {LABEL['oracle']} (pp)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        save_plot(figname=f"cifar10c_severity_{filename}")


def plot_cifar10c_optimal_depth(filename: str, save: bool = False) -> None:
    r"""
    Plot optimal Cifar10-C probing depth depending on the corruption severity.

    Parameters
    ----------
    filename: str
        Name of the analysis result file, without its ``.csv`` suffix.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 5.5
    height = 5
    figsize = width, height
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)].copy()
    df["depth"] = 100 * df["best_layer"] / (df["n_layers"] - 1)

    severity_order = [1, 3, 5]
    sns.barplot(
        data=df,
        x="severity",
        y="depth",
        order=severity_order,
        estimator=np.mean,
        errorbar=("ci", 95),
        color=GAIN_PALETTE["oracle"],
        legend=False,
        err_kws={
            "color": "#333333",
            "linewidth": 0.5,
            "linestyle": "--",
        },
        capsize=0.15,
        ax=ax,
    )
    ax.grid(axis="y", alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_ylim(82, 96)
    ax.set_yticks([82, 89, 96])
    ax.set_xlabel("Corruption severity", fontsize=FONTSIZE)
    ax.set_ylabel(r"Optimal layer depth ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)
    plt.tight_layout()
    if save:
        save_plot(figname=f"cifar10c_optimal_depth_{filename}")


def plot_pretraining_scale_vision(filename: str, save: bool = False) -> None:
    r"""
    Plot the average oracle gain over standard probing for different pretraining scales,
    as a barplot with 95% confidence intervals computed over all (model, dataset) pairs
    within each family.

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
    # Figure
    width = 5.5
    height = 5
    figsize = width, height
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = df[df["dataset_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Compute error reduction and relative gain
    df["error_reduction"] = 100 * df["gain"] / (100 - df["baseline_accuracy"])
    print(f"error reduction: {df.groupby('model_name')['error_reduction'].mean()}")

    # Results
    pretraining_labels = {"vit": r"$0.25$T", "dinov2": r"$1.83$T", "aimv2": r"$3.15$T", "dinov3": r"$3.90$T"}
    family_order = ["vit", "dinov2", "aimv2", "dinov3"]
    sns.barplot(
        data=df,
        x="model_name",
        y="gain",
        order=family_order,
        estimator=np.mean,
        errorbar=("ci", 95),
        color=GAIN_PALETTE["oracle"],
        legend=False,
        err_kws={
            "color": "#333333",
            "linewidth": 0.5,
            "linestyle": "--",
        },
        capsize=0.15,
        ax=ax,
    )

    # Annotate model family name above each bar
    bar_heights = [df[df["model_name"] == m]["gain"].mean() for m in family_order]
    for i, (model_name, height) in enumerate(zip(family_order, bar_heights, strict=True)):
        xytext = (0, 18)
        ax.annotate(
            VFM_NAMES[model_name],
            xy=(i, height),
            xytext=xytext,
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=18,
            color="#333333",
            zorder=10,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
        )

    # Visualization
    ax.grid(axis="y", alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    yticks = [0, 2.5, 5]
    ax.set_yticks(yticks)
    ax.set_xticklabels(list(pretraining_labels.values()))
    ax.set_xlabel("Number of pretraining tokens", fontsize=FONTSIZE)
    ax.set_ylabel(rf"Oracle gain {LABEL['oracle']} (pp)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        figname = f"pretraining_scale_vision_{filename}"
        save_plot(figname=figname)


def plot_model_family_vision(filename: str, save: bool = False, ncol: int = 1) -> None:
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
    ncol: int, default=1
        Number of columns in the legend.
    """
    # Figure
    width = 5.5
    height = 4.4
    ncols = len(REPRESENTATIVE_DATASETS)
    figsize = ncols * width, height
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize, sharey=True)

    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = df[df["dataset_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_VFMs.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = df[mask]

    # Results
    model_order = list(REPRESENTATIVE_VFMs.keys())
    gain_types = ["gain", "gain_module", "gain_layer"]
    gain_labels = ["oracle", "module", "layer"]
    for i, ax in enumerate(axes):
        dataset = REPRESENTATIVE_DATASETS[i]
        sub = df[df["dataset_name"] == dataset]
        agg = sub.groupby("model_name")[gain_types].mean().reindex(model_order)
        agg_long = agg.reset_index().melt(
            id_vars="model_name",
            value_vars=gain_types,
            var_name="gain_type",
            value_name="gain_value",
        )
        agg_long["gain_type"] = agg_long["gain_type"].map(dict(zip(gain_types, gain_labels, strict=False)))
        sns.barplot(
            data=agg_long,
            x="model_name",
            y="gain_value",
            hue="gain_type",
            order=model_order,
            hue_order=gain_labels,
            palette=GAIN_PALETTE,
            err_kws={"color": "#333333", "linewidth": 1},
            capsize=0.15,
            ax=ax,
        )

        # Visualization
        ax.grid(axis="y", alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1)
        ax.legend().remove()
        ax.set_title(DATASETS[dataset], fontsize=FONTSIZE)
        ax.set_xlabel("")
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels([VFM_NAMES[model] for model in model_order])
        if i == 0:
            ax.set_ylabel("Gain over standard probing (pp)", fontsize=FONTSIZE)
    sns.despine(fig, axes[0], trim=True, right=True, offset=10)

    # Common legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    labels = [LABEL[key] for key in gain_labels]
    legend = fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.3, 0.78),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=True,
        fontsize=LEG_FONTSIZE,
    )
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(10)
    plt.tight_layout()
    if save:
        figname = f"model_family_vision_{filename}"
        save_plot(figname=figname)


def plot_probing_depth_vision(filename: str, save: bool = False) -> None:
    r"""
    Evolution of the act and rc2 probing accuracy along with the depth.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 5
    height = 4
    panel_models = (("vit", "dinov3"), ("dinov2", "aimv2"))
    nrows = len(panel_models[0])
    ncols = len(REPRESENTATIVE_DATASETS)
    figsize = 2 * ncols * width, nrows * height
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        nrows=nrows,
        ncols=2 * ncols + 1,
        width_ratios=[1] * ncols + [0.2] + [1] * ncols,
        wspace=0.2,
    )
    axes = np.empty((nrows, 2 * ncols), dtype=object)
    reference = None
    for i in range(nrows):
        for column in range(2 * ncols):
            grid_column = column if column < ncols else column + 1
            axes[i, column] = fig.add_subplot(grid[i, grid_column], sharex=reference)
            if reference is None:
                reference = axes[i, column]

    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = df[df["dataset_name"].str.contains(pattern, case=False, na=False)]

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_VFMs.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = df[mask]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    for panel, model_names in enumerate(panel_models):
        for i, model_name in enumerate(model_names):
            for j, dataset_name in enumerate(REPRESENTATIVE_DATASETS):
                column = panel * ncols + j
                ax = axes[i, column]
                model_size = REPRESENTATIVE_VFMs[model_name]
                for module in ["rc2", "act"]:
                    subset = df[
                        (df["model_name"] == model_name)
                        & (df["model_size"] == model_size)
                        & (df["dataset_name"] == dataset_name)
                        & (df["module"] == module)
                    ].copy()
                    subset = subset.sort_values("layer")
                    n_layers = subset["layer"].max()
                    subset["depth"] = 100 * subset["layer"] / n_layers
                    ax.plot(
                        subset["depth"],
                        subset["accuracy"],
                        color=MODULE_PALETTE[module],
                        linewidth=LINEWIDTH,
                        linestyle="-" if module != "rc2" else "--",
                        label=module.upper() if module != "act" else "Act",
                    )

                    # Mark the highest act performance
                    if module == "act":
                        index = np.argmax(subset["accuracy"])
                        ax.scatter(
                            subset["depth"].values[index],
                            subset["accuracy"].values[index],
                            label="Best Act",
                            edgecolor="#333333",
                            s=300,
                            c=MODULE_PALETTE[module],
                            marker="*",
                            zorder=3,
                        )

                # Visualization
                ax.grid(alpha=ALPHA_GRID, lw=1.3)
                ax.spines["left"].set_linewidth(1)
                ax.spines["right"].set_linewidth(1)
                ax.spines["top"].set_linewidth(1)
                ax.spines["bottom"].set_linewidth(1)
                ax.tick_params(
                    axis="both",
                    direction="out",
                    length=5,
                    width=1,
                    labelsize=FONTSIZE,
                    labelbottom=i == nrows - 1,
                )
                ymin, ymax = ax.get_ylim()
                tol = 1
                ax.set_yticks(np.linspace(ymin - tol, ymax - tol, 3).astype(int))
                if i == nrows - 1:
                    ax.set_xlabel(r"Layer depth ($\%$)", fontsize=FONTSIZE)
                if i == 0:
                    ax.set_title(f"{DATASETS[dataset_name]}\n", fontsize=FONTSIZE)
                if j == 0:
                    ax.set_ylabel(f"{VFM_NAMES[model_name]}\n" + r"Accuracy ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, axes[0, 0], trim=True, right=True, offset=10)

    # Common legend
    legend = axes[0, 0].legend(
        loc="upper center",
        bbox_to_anchor=(0.63, 0.52),
        fancybox=True,
        borderaxespad=0,
        shadow=False,
        frameon=True,
        handlelength=1.9,
        fontsize=LEG_FONTSIZE,
    )
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(10)

    # Double legend
    legend_bis = axes[0, 3].legend(
        loc="upper center",
        bbox_to_anchor=(0.63, 0.52),
        fancybox=True,
        borderaxespad=0,
        shadow=False,
        frameon=True,
        handlelength=1.9,
        fontsize=LEG_FONTSIZE,
    )
    legend_bis.get_frame().set_edgecolor("none")
    legend_bis.get_frame().set_facecolor("white")
    legend_bis.get_frame().set_alpha(1.0)
    legend_bis.set_zorder(10)
    plt.tight_layout()
    if save:
        figname = f"probing_depth_vision_{filename}"
        save_plot(figname=figname)


def plot_probing_depth_family_vision(filename: str, model_name: str, save: bool = False) -> None:
    r"""
    Evolution of the act and rc2 probing accuracy along with the depth.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 5
    height = 4
    sizes = VFM_SIZES[model_name]
    nrows = len(sizes)
    ncols = len(REPRESENTATIVE_DATASETS)
    figsize = ncols * width, nrows * height
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)

    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join([model_name])
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = df[df["dataset_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    model_labels = {
        "vit": ["ViT-B", "ViT-L", "ViT-H"],
        "dinov2": ["DINOv2-S", "DINOv2-B", "DINOv2-L", "DINOv2-g"],
        "dinov3": ["DINOv3-S", "DINOv3-B", "DINOv3-L", "DINOv3-7B"],
        "aimv2": ["AIMv2-L", "AIMv2-H", "AIMv2-1B", "AIMv2-3B"],
    }
    for i, model_size in enumerate(sizes):
        for j, dataset_name in enumerate(REPRESENTATIVE_DATASETS):
            ax = axes[i, j]
            for module in ["act", "rc2"]:
                subset = df[
                    (df["model_name"] == model_name)
                    & (df["model_size"] == model_size)
                    & (df["dataset_name"] == dataset_name)
                    & (df["module"] == module)
                ].copy()
                subset = subset.sort_values("layer")
                n_layers = subset["layer"].max()
                subset["depth"] = 100 * subset["layer"] / n_layers
                ax.plot(
                    subset["depth"],
                    subset["accuracy"],
                    color=MODULE_PALETTE[module],
                    linewidth=LINEWIDTH,
                    linestyle="-" if module != "rc2" else "--",
                    label=module.upper() if module != "act" else "Act",
                )

                # Mark the highest act performance
                if module == "act":
                    index = np.argmax(subset["accuracy"])
                    ax.scatter(
                        subset["depth"].values[index],
                        subset["accuracy"].values[index],
                        label="Best Act",
                        edgecolor="#333333",
                        s=300,
                        c=MODULE_PALETTE[module],
                        marker="*",
                        zorder=3,
                    )

            # Visualization
            ax.grid(alpha=ALPHA_GRID, lw=1.3)
            ax.spines["left"].set_linewidth(1)
            ax.spines["right"].set_linewidth(1)
            ax.spines["top"].set_linewidth(1)
            ax.spines["bottom"].set_linewidth(1)
            ax.tick_params(axis="both", direction="out", length=5, width=1, labelsize=FONTSIZE)
            ymin, ymax = ax.get_ylim()
            tol = 1
            ax.set_yticks(np.linspace(ymin - tol, ymax - tol, 3).astype(int))
            if i == 0:
                ax.set_title(f"{DATASETS[dataset_name]}\n", fontsize=FONTSIZE)
            if i == nrows - 1:
                ax.set_xlabel(r"Layer depth ($\%$)", fontsize=FONTSIZE)
            if j == 0:
                ax.set_ylabel(f"{model_labels[model_name][i]}\n" + r"Accuracy ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, axes[0, 0], trim=True, right=True, offset=10)

    # Common legend
    legend = axes[0, 0].legend(
        loc="upper center",
        bbox_to_anchor=(0.63, 0.52),
        fancybox=True,
        borderaxespad=0,
        shadow=False,
        frameon=True,
        handlelength=1.9,
        fontsize=LEG_FONTSIZE,
    )
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(10)
    plt.tight_layout()
    if save:
        figname = f"probing_depth_family_{model_name}_vision_{filename}"
        save_plot(figname=figname)


def plot_heatmap_winning_frequency_vision(filename: str, save: bool = False) -> None:
    r"""
    Plot the distribution of winning layer-module locations for vision models.

    Each model-dataset pair contributes one vote to its jointly optimal
    layer-module pair. Relative depth is aggregated into configurable bins
    for both the heatmap and the depth marginal.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    n_modules = 8
    depth_step = 5
    n_bins = 100 // depth_step
    width = 10
    height = 4
    figsize = width, height
    fig = plt.figure(figsize=figsize)
    bar_color = "#c6a4e4"
    cmap = sns.cubehelix_palette(
        # start=0,
        rot=0.2,
        dark=0.45,
        light=0.95,
        as_cmap=True,
    )
    heatmap_ratio = n_bins / n_modules
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[heatmap_ratio, 0.55, 0.06],
        height_ratios=[0.45, 1.0],
        wspace=0.05,
        hspace=0.05,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = df[df["dataset_name"].str.contains(pattern, case=False, na=False)]

    # Results
    modules = ["ln1", "mha", "rc1", "ln2", "fc1", "act", "fc2", "rc2"]
    module_labels = {
        "ln1": r"$\mathtt{LN1}$",
        "mha": r"$\mathtt{MHA}$",
        "rc1": r"$\mathtt{RC1}$",
        "ln2": r"$\mathtt{LN2}$",
        "fc1": r"$\mathtt{FC1}$",
        "act": r"$\mathtt{Act}$",
        "fc2": r"$\mathtt{FC2}$",
        "rc2": r"$\mathtt{RC2}$",
    }

    best_depth = 100 * df["best_layer"].to_numpy(dtype=float) / (df["n_layers"].to_numpy(dtype=float) - 1)
    depth_bins = np.arange(0, 100 + depth_step, depth_step, dtype=float)
    n_bins = len(depth_bins) - 1
    depth_bin = np.minimum(
        (best_depth / depth_step).astype(int),
        n_bins - 1,
    )
    pair_counts = np.zeros((n_bins, len(modules)), dtype=float)
    module_to_idx = {module: i for i, module in enumerate(modules)}

    for bin_idx, module in zip(
        depth_bin,
        df["best_module"].to_numpy(dtype=str),
        strict=True,
    ):
        module_idx = module_to_idx.get(module)
        if module_idx is not None:
            pair_counts[bin_idx, module_idx] += 1

    n_pairs = len(df)
    pair_pct = 100 * pair_counts / n_pairs
    module_pct = pair_pct.sum(axis=0)
    depth_pct = pair_pct.sum(axis=1)

    print("\nOptimal module frequency:")
    for module, frequency in sorted(
        zip(modules, module_pct, strict=True),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {module:>4s}: {frequency:.1f}%")

    print("\nOptimal depth frequency:")
    for i, frequency in enumerate(depth_pct):
        if frequency > 0:
            lower = depth_bins[i]
            upper = depth_bins[i + 1]
            print(f"  {lower:.0f}–{upper:.0f}%: {frequency:.1f}%")

    # Results
    heatmap_edges = np.arange(n_bins + 1)
    module_edges = np.arange(len(modules) + 1)
    im = ax_heatmap.pcolormesh(heatmap_edges, module_edges, pair_pct.T, shading="flat", cmap=cmap, edgecolors="none")
    depth_centers = np.arange(n_bins) + 0.5
    ax_top.bar(depth_centers, depth_pct, color=bar_color, width=1.0)
    module_centers = np.arange(len(modules)) + 0.5
    ax_right.barh(module_centers, module_pct, color=bar_color, height=1.0)

    # Visualization heatmap
    ax = ax_heatmap
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(
        axis="both",
        direction="out",
        length=5,
        width=1,
        labelsize=FONTSIZE,
    )
    ax.set_xlim(0, n_bins)
    ax.set_ylim(0, len(modules))
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_yticks(module_centers)
    ax.set_yticklabels([module_labels[module] for module in modules])
    ax.set_xlabel("Layer depth (%)")
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Visualiation colorbar
    cbar = fig.colorbar(im, cax=ax_cbar, ticks=np.linspace(pair_pct.min(), pair_pct.max(), 3))
    cbar.set_label("Optimality frequency (%)", fontsize=FONTSIZE, rotation=270, labelpad=25)
    ax = ax_cbar
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1, labelsize=FONTSIZE)
    ax.set_yticks([0, 13, 26])

    # Visualization top marginal
    ax = ax_top
    ax.set_xlim(0, n_bins)
    ax.set_xticks([])
    ax.set_yticks([])

    best_idx = np.argmax(depth_pct)
    ax.text(
        depth_centers[best_idx],
        depth_pct[best_idx] + 1,
        f"{depth_pct[best_idx]:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        color="black",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Visualization right marginal
    ax = ax_right
    ax.set_ylim(0, len(modules))
    ax.set_xlim(0, module_pct.max() * 1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    best_idx = np.argmax(module_pct)
    ax.text(
        module_pct[best_idx] + 1,
        module_centers[best_idx],
        f"{module_pct[best_idx]:.1f}%",
        ha="left",
        va="center",
        fontsize=12,
        color="black",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Save
    if save:
        save_plot(figname=f"heatmap_winning_frequency_vision_{filename}")


def plot_heatmap_winning_frequency_language(
    filename: str,
    save: bool = False,
) -> None:
    r"""
    Plot the distribution of winning layer-module locations for language models.

    Each model contributes one vote to its jointly optimal layer-module pair.
    Relative depth is aggregated into configurable bins for both the heatmap
    and the depth marginal.

    Parameters
    ----------
    filename : str
        Name of the result file to use.
    save : bool, default=False
        Whether to save the plot.
    """
    # --------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------
    depth_step = 10  # Relative depth bin size (%) for heatmap and marginal

    # --------------------------------------------------------------------------
    # Figure
    # --------------------------------------------------------------------------
    width = 6.5
    height = 6.0
    fig = plt.figure(figsize=(width, height))

    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.0, 0.20, 0.06],
        height_ratios=[0.20, 1.0],
        wspace=0.05,
        hspace=0.05,
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # --------------------------------------------------------------------------
    # Recover aggregate results
    # --------------------------------------------------------------------------
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep language models
    pattern = "|".join(LLMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # --------------------------------------------------------------------------
    # Relative depth
    # --------------------------------------------------------------------------
    df["best_depth"] = 100 * df["best_layer"] / (df["n_layers"] - 1)

    # --------------------------------------------------------------------------
    # Fixed module order
    # Keep all modules, even if some are never selected.
    # --------------------------------------------------------------------------
    modules = [
        "ln1",
        "mha",
        "rc1",
        "ln2",
        "fc1",
        "act",
        "fc2",
        "rc2",
    ]

    # --------------------------------------------------------------------------
    # Bin relative depth
    # --------------------------------------------------------------------------
    depth_bins = np.arange(
        0,
        100 + depth_step,
        depth_step,
    )

    df["depth_bin"] = pd.cut(
        df["best_depth"],
        bins=depth_bins,
        labels=False,
        include_lowest=True,
        right=True,
    )

    n_bins = len(depth_bins) - 1

    # --------------------------------------------------------------------------
    # Count exact best depth-module pairs
    # --------------------------------------------------------------------------
    pair_counts = df.groupby(["depth_bin", "best_module"], observed=False).size().unstack(fill_value=0)

    # Keep all depth bins and all modules.
    # Modules not selected by any model receive zero.
    pair_counts = pair_counts.reindex(
        index=np.arange(n_bins),
        columns=modules,
        fill_value=0,
    )

    n_runs = len(df)

    # Convert to percentage of model runs
    pair_pct = 100 * pair_counts / n_runs

    # --------------------------------------------------------------------------
    # Marginals
    # --------------------------------------------------------------------------
    module_pct = pair_pct.sum(axis=0)
    depth_pct = pair_pct.sum(axis=1)

    # Print frequencies
    print("\nOptimal module frequency:")
    for module, frequency in module_pct.items():
        print(f"  {module:>4s}: {frequency:.1f}%")

    print("\nOptimal depth frequency:")
    for i, frequency in depth_pct.items():
        if frequency > 0:
            lower = depth_bins[i]
            upper = depth_bins[i + 1]
            print(f"  {lower:.0f}–{upper:.0f}%: {frequency:.1f}%")

    # --------------------------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------------------------
    im = ax_heatmap.imshow(
        pair_pct.values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, len(modules), 0, 100],
    )

    # Make the heatmap slightly taller than wide
    ax_heatmap.set_box_aspect(1.0)

    # Module axis
    ax_heatmap.set_xticks(
        np.arange(len(modules)) + 0.5,
    )
    ax_heatmap.set_xticklabels(modules)

    # Depth axis
    ax_heatmap.set_yticks([0, 25, 50, 75, 100])
    ax_heatmap.set_yticklabels(["0", "25", "50", "75", "100"])
    ax_heatmap.set_ylabel("Layer depth (%)")

    # Clean up spines
    ax_heatmap.spines["top"].set_visible(False)
    ax_heatmap.spines["right"].set_visible(False)

    # --------------------------------------------------------------------------
    # Top marginal: module frequency
    # --------------------------------------------------------------------------
    x = np.arange(len(modules))

    ax_top.bar(
        x + 0.5,
        module_pct.values,
        width=1.0,
    )

    ax_top.set_xlim(0, len(modules))
    ax_top.set_xticks([])
    ax_top.set_yticks([])

    for spine in ax_top.spines.values():
        spine.set_visible(False)

    # Annotate the most frequent module
    best_module_idx = int(np.argmax(module_pct.values))
    best_module_frequency = module_pct.iloc[best_module_idx]

    ax_top.text(
        best_module_idx + 0.5,
        best_module_frequency,
        f"{best_module_frequency:.1f}%",
        ha="center",
        va="bottom",
    )

    # --------------------------------------------------------------------------
    # Right marginal: depth frequency
    # --------------------------------------------------------------------------
    y = np.arange(n_bins)

    ax_right.barh(
        y + 0.5,
        depth_pct.values,
        height=1.0,
    )

    ax_right.set_ylim(0, n_bins)
    ax_right.set_xlim(0, depth_pct.max() * 1.15)
    ax_right.set_xticks([])
    ax_right.set_yticks([])

    for spine in ax_right.spines.values():
        spine.set_visible(False)

    # --------------------------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------------------------
    cbar = fig.colorbar(
        im,
        cax=ax_cbar,
    )
    cbar.set_label("Frequency (%)")

    # --------------------------------------------------------------------------
    # Save
    # --------------------------------------------------------------------------
    if save:
        save_plot(figname=f"heatmap_winning_frequency_language_{filename}")


def plot_winning_distribution_vision_old(filename: str, save: bool = False) -> None:
    r"""
    Plot the distribution of winning layer-module locations for vision models.

    Each evaluation contributes one vote, split equally across tied winners.
    The x-axis is normalized layer depth, the y-axis is the module, and the
    bar height is the percentage of evaluations won by that location.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 5.5
    height = 5
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111, projection="3d")

    # Recover results
    df = pd.read_csv(RESULT_DIR / f"{filename}.csv")
    df = df[df["model_name"].str.lower().isin(VFM_NAMES)].copy()
    pattern = "|".join(DATASETS.keys())

    df = df[df["dataset_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Recover winning locations. Ties split one vote equally between winners.
    group_cols = ["dataset_name", "model_name", "model_size"]
    grouped = df.groupby(group_cols, observed=True)
    last_layer = grouped["layer"].transform("max")
    oracle_accuracy = grouped["accuracy"].transform("max")
    winners = df.loc[np.isclose(df["accuracy"], oracle_accuracy, atol=1e-8, rtol=0)].copy()
    winners["depth_pct"] = 100 * winners["layer"] / last_layer.loc[winners.index].clip(lower=1)
    n_winners = winners.groupby(group_cols, observed=True)["module"].transform("size")
    winners["weight"] = 1 / n_winners
    n_units = int(grouped.ngroups)

    # Bin depth to make the winning mass easy to read.
    depth_bin = 5
    depth_edges = np.arange(0, 100 + depth_bin, depth_bin, dtype=float)
    depth_left = depth_edges[:-1]
    bin_index = np.clip(
        np.digitize(winners["depth_pct"], depth_edges, right=False) - 1,
        0,
        len(depth_left) - 1,
    )
    module_index = winners["module"].map({module: i for i, module in enumerate(MODULE_ORDER)}).to_numpy()
    grid = np.zeros((len(MODULE_ORDER), len(depth_left)), dtype=float)
    np.add.at(grid, (module_index, bin_index), 100 * winners["weight"].to_numpy() / n_units)

    # Draw one consistent color per module; height, not color, encodes frequency.
    depth_width = depth_bin * 0.78
    module_width = 0.38
    for module_idx, module in enumerate(MODULE_ORDER):
        columns = np.flatnonzero(grid[module_idx] > 0)
        if len(columns) == 0:
            continue
        alpha = 1.0 if module == "act" else 0.72
        ax.bar3d(
            np.full(len(columns), module_idx - module_width / 2),
            depth_left[columns],
            np.zeros(len(columns)),
            np.full(len(columns), module_width),
            np.full(len(columns), depth_width),
            grid[module_idx, columns],
            color=MODULE_PALETTE[module],
            alpha=alpha,
            edgecolor="white",
            linewidth=0.25,
            shade=True,
        )

    # Mark the single most frequent winning location.
    best_module_idx, best_depth_idx = np.unravel_index(np.argmax(grid), grid.shape)
    best_value = grid[best_module_idx, best_depth_idx]
    if best_value > 0:
        ax.scatter(
            best_module_idx,
            depth_left[best_depth_idx] + depth_width / 2,
            best_value + 1.5,
            marker="*",
            s=320,
            color="#d62728",
            edgecolor="white",
            linewidth=0.8,
            depthshade=False,
            zorder=10,
        )

    # Visualization
    module_labels = [module.upper() if module != "act" else "Act" for module in MODULE_ORDER]
    z_tick_max = max(10, 10 * int(np.ceil(grid.max() / 10)))
    z_limit = z_tick_max * 1.1
    ax.set(
        xlabel="\nModule",
        ylabel="\nLayer depth (%)",
        zlabel="\nWinning frequency (%)",
        xlim=(-0.6, len(MODULE_ORDER) - 0.4),
        ylim=(0, 100),
        zlim=(0, z_limit),
        xticks=np.arange(len(MODULE_ORDER)),
        xticklabels=module_labels,
        yticks=[0, 25, 50, 75, 100],
    )
    ax.set_zticks(np.arange(0, z_tick_max + 1, 10))
    ax.view_init(elev=25, azim=40)
    ax.set_box_aspect((2.5, 1.6, 1.8))
    ax.tick_params(axis="both", direction="out", length=5, width=1, labelsize=FONTSIZE - 3, pad=6)
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="z", labelsize=FONTSIZE - 3, pad=4)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.96, 0.96, 0.96, 1.0))
        axis.pane.set_edgecolor("white")
    ax.set_title("Winning layer–module locations", fontsize=FONTSIZE, pad=18)
    fig.set_layout_engine("none")

    if save:
        save_plot(figname=f"winning_distribution_vision_{filename}")


def plot_oracle_recovery_vision(filename: str, save: bool = False) -> None:
    r"""
    Recover the table of oracle gain recovery (computed as gain / oracle_gain) averaged over all datasets and models.

    Strategies:
        - Best layer on rc2
        - Fixed (last layer on act)
        - Best layer on act

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 5
    height = 4
    figsize = width, height
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = df[df["dataset_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    order = ["Best RC2", "Fixed", "Best ACT"]
    groups = ["dataset_name", "model_name", "model_size"]
    rows = []
    for key, sub in df.groupby(groups):
        last_layer = sub["layer"].max()

        # Standard
        baseline = sub[(sub.layer == last_layer) & (sub.module == "rc2")].accuracy.iloc[0]

        # Best RC2
        rc2_best = sub[sub.module == "rc2"].accuracy.max()

        # Fixed: last layer act
        heuristic = sub[(sub.layer == last_layer) & (sub.module == "act")].accuracy.iloc[0]

        # Best act
        act_best = sub[sub.module == "act"].accuracy.max()

        # Recover strategies
        strategies = {
            "Best RC2": rc2_best,
            "Fixed": heuristic,
            "Best ACT": act_best,
        }

        # Recover oracle recovery
        oracle = sub.accuracy.max()
        oracle_gain = oracle - baseline

        # Ignore non zero oracle gains
        if oracle_gain > 0:
            for strategy, acc in strategies.items():
                # If a strategy does not provide any improvement, we assume it recovers 0% of the oracle gain
                gain = np.max((acc - baseline, 0))
                recovery = gain / oracle_gain if gain > 0 else 0
                rows.append(
                    {
                        "dataset": key[0],
                        "family": key[1],
                        "size": key[2],
                        "strategy": strategy,
                        "recovery": recovery * 100,
                    }
                )
    summary = pd.DataFrame(rows)
    means = summary.groupby("strategy")["recovery"].mean().reindex(order)
    print(means)
    sns.barplot(
        data=summary,
        x="strategy",
        y="recovery",
        order=order,
        estimator=np.mean,
        errorbar=("ci", 95),
        palette=[MODULE_PALETTE["rc2"], MODULE_PALETTE["act"], MODULE_PALETTE["act"]],
        legend=False,
        err_kws={
            "color": "#333333",
            "linewidth": 0.5,
            "linestyle": "--",
        },
        capsize=0.15,
        ax=ax,
    )
    heuristic_idx = order.index("Fixed")
    bar = ax.patches[heuristic_idx]
    bar.set_hatch("//")

    # Visualization
    ax.grid(axis="y", alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xticklabels(["Best RC2", r"$\mathtt{Act@100}$", "Best Act"])
    ax.set_ylim(15, 85)
    ax.set_yticks([15, 50, 85])
    ax.set_xlabel("")
    ax.set_ylabel(r"Oracle gain recovered ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        save_plot(figname=f"oracle_recovery_vision_{filename}")


# VFMs table functions
# ------------------------------------------------------------------------------
def table_model_size_vision(filename: str) -> None:
    r"""
    Friedman test to determine whether the model size has a significant impact on the gain
    per model family. The null hypothesis is that model size does not influence the gain.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    """
    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = df[df["dataset_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    results = []
    for family, sizes in VFM_SIZES.items():
        sub = df[df["model_name"] == family]

        # Build matrix: rows=datasets, cols=sizes
        pivot = (
            sub.pivot_table(index="dataset_name", columns="model_size", values="gain").reindex(columns=sizes).dropna()
        )

        # Friedman test
        groups = [pivot[s].values for s in pivot.columns]
        stat, p = friedmanchisquare(*groups)

        # Kendall's W (effect size)
        k = len(pivot.columns)
        n = len(pivot)
        W = stat / (n * (k - 1))
        results.append(
            {
                "Family": family,
                "chi2": stat,
                "p-value": p,
                "Kendall's W": W,
            }
        )

    friedman_table = pd.DataFrame(results)
    print(friedman_table.to_latex(index=False, float_format="%.3f"))


def table_winning_module_frequency_vision(filename: str) -> None:
    r"""
    Frequency with which each transformer module is selected
    as the optimal probing module over all vision experiments.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    """
    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep only vision foundation models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = df[df["dataset_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    freq = df["best_module"].value_counts(normalize=True).mul(100)
    freq = freq.reindex(MODULE_ORDER, fill_value=0.0)

    table = freq.to_frame().T
    table.index = ["Winning freq. (%)"]
    print(table.to_latex(index=True, float_format="%.1f"))


def table_lodo_depth_vision(filename: str) -> None:
    r"""
    Leave-one-dataset-out selection of the fixed vision probing depth.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    """
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = df[df["dataset_name"].str.contains(pattern, case=False, na=False)].copy()

    # LODO
    grid_step = 5
    datasets = df["dataset_name"].unique()
    candidate_depths = np.arange(0, 101, grid_step)
    model_keys = ["model_name", "model_size"]
    last_layers = df.groupby(model_keys)["layer"].max()
    rows = []
    for held_out in datasets:
        train = df[df["dataset_name"] != held_out]
        scores = []
        for depth in candidate_depths:
            dataset_scores = []
            for _, dataset in train[train["module"] == "act"].groupby("dataset_name"):
                model_scores = []
                for model_key, model in dataset.groupby(model_keys):
                    layer = int(depth / 100 * last_layers[model_key])
                    model_scores.append(model[model["layer"] == layer]["accuracy"].iloc[0])
                dataset_scores.append(np.mean(model_scores))
            scores.append(np.mean(dataset_scores))
        train_score, selected_depth = max(
            zip(scores, candidate_depths, strict=False),
            key=lambda item: (item[0], item[1]),
        )

        held_out_scores = []
        for model_key, model in df[(df["dataset_name"] == held_out) & (df["module"] == "act")].groupby(model_keys):
            layer = int(selected_depth / 100 * last_layers[model_key])
            held_out_scores.append(model[model["layer"] == layer]["accuracy"].iloc[0])
        rows.append(
            {
                "dataset": held_out,
                "selected_depth": selected_depth,
                "train_score": train_score,
                "held_out_accuracy": np.mean(held_out_scores),
            }
        )

    table = pd.DataFrame(rows).sort_values("dataset")
    choice = table["selected_depth"].mode().iloc[0]
    print(f"LODO selected depth: {choice}%")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


def table_performance_comparison_vision(filename: str) -> None:
    r"""
    Performance comparison between standard probing, heuristic probing and oracle probing.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    """
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep only vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_VFMs.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = df[mask]

    # Results
    rows = []
    groups = ["dataset_name", "model_name"]
    for (dataset, family), sub in df.groupby(groups):
        last_layer = sub.layer.max()
        standard = sub[(sub.layer == last_layer) & (sub.module == "rc2")].accuracy.iloc[0]
        heuristic = sub[(sub.layer == last_layer) & (sub.module == "act")].accuracy.iloc[0]
        oracle = sub.accuracy.max()
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "family": family,
                    "strategy": "standard",
                    "accuracy": standard,
                },
                {
                    "dataset": dataset,
                    "family": family,
                    "strategy": "heuristic",
                    "accuracy": heuristic,
                },
                {
                    "dataset": dataset,
                    "family": family,
                    "strategy": "oracle",
                    "accuracy": oracle,
                },
            ]
        )
    summary = pd.DataFrame(rows)
    table = summary.groupby(["family", "strategy", "dataset"])["accuracy"].mean().unstack("dataset")
    table["Average"] = table.mean(axis=1)

    # Overall rows across all models, sizes, and datasets
    overall_rows = summary.groupby(["strategy", "dataset"])["accuracy"].mean().unstack("dataset")
    overall_rows["Average"] = overall_rows.mean(axis=1)

    # Latex table
    print(table.to_latex(float_format="%.1f", multirow=True, bold_rows=True))


def table_performance_comparison_all_vision(filename: str) -> None:
    r"""
    Performance comparison across all model sizes.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    """
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep only vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Results
    rows = []
    groups = ["dataset_name", "model_name", "model_size"]
    for (dataset, family, size), sub in df.groupby(groups):
        last_layer = sub.layer.max()
        standard = sub[(sub.layer == last_layer) & (sub.module == "rc2")].accuracy.iloc[0]
        heuristic = sub[(sub.layer == last_layer) & (sub.module == "act")].accuracy.iloc[0]
        oracle = sub.accuracy.max()
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "family": family,
                    "size": size,
                    "strategy": "standard",
                    "accuracy": standard,
                },
                {
                    "dataset": dataset,
                    "family": family,
                    "size": size,
                    "strategy": "heuristic",
                    "accuracy": heuristic,
                },
                {
                    "dataset": dataset,
                    "family": family,
                    "size": size,
                    "strategy": "oracle",
                    "accuracy": oracle,
                },
            ]
        )
    summary = pd.DataFrame(rows)
    table = summary.groupby(["family", "size", "strategy", "dataset"])["accuracy"].mean().unstack("dataset")
    table["Average"] = table.mean(axis=1)

    # Overall rows across all models, sizes, and datasets
    overall_rows = summary.groupby(["strategy", "dataset"])["accuracy"].mean().unstack("dataset")
    overall_rows["Average"] = overall_rows.mean(axis=1)

    # Latex table
    print(table.to_latex(float_format="%.1f", multirow=True, bold_rows=True))


def table_global_statistics_vision(filename: str) -> None:
    r"""
    Compute win rates for fixed and depth-optimized vision strategies.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    """
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep vision models and datasets
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    pattern = "|".join(DATASETS.keys())
    df = df[df["dataset_name"].str.contains(pattern, case=False, na=False)]

    def print_statistics(frame: pd.DataFrame, scope: str) -> None:
        rows = []
        for _, sub in frame.groupby(["dataset_name", "model_name", "model_size"]):
            last_layer = sub["layer"].max()
            rows.append(
                {
                    "standard": sub[(sub["layer"] == last_layer) & (sub["module"] == "rc2")]["accuracy"].iloc[0],
                    "fixed": sub[(sub["layer"] == last_layer) & (sub["module"] == "act")]["accuracy"].iloc[0],
                    "best_rc2": sub[sub["module"] == "rc2"]["accuracy"].max(),
                    "best_act": sub[sub["module"] == "act"]["accuracy"].max(),
                }
            )
        evaluations = pd.DataFrame(rows)
        print(f"\n{scope} vision evaluations: {len(evaluations)}")
        for strategy, label in [
            ("fixed", "Fixed ACT@100%"),
            ("best_rc2", "Best depth on RC2"),
            ("best_act", "Best depth on ACT"),
        ]:
            wins = (evaluations[strategy] > evaluations["standard"]).sum()
            gain = (evaluations[strategy] - evaluations["standard"]).mean()
            print(f"{label}: {wins}/{len(evaluations)} ({100 * wins / len(evaluations):.1f}%)")
            print(f"{label} average gain: {gain:.2f} pp")

    print_statistics(df, "All model sizes")
    large_df = df[df["model_size"] == df["model_name"].map(REPRESENTATIVE_VFMs)]
    print_statistics(large_df, "Large variants")


def table_global_statistics_ood_vision(filename: str) -> None:
    r"""
    Compute win rates for fixed and depth-optimized strategies on hard vision datasets.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    """
    hard_datasets = [
        "iwildcam",
        "rxrx1",
        "domainnet-clipart",
        "domainnet-infograph",
        "domainnet-painting",
        "domainnet-quickdraw",
        "domainnet-sketch",
    ]

    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep vision models and hard datasets
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    df = df[df["dataset_name"].isin(hard_datasets)]

    def print_statistics(frame: pd.DataFrame, scope: str) -> None:
        rows = []
        for _, sub in frame.groupby(["dataset_name", "model_name", "model_size"]):
            last_layer = sub["layer"].max()
            rows.append(
                {
                    "standard": sub[(sub["layer"] == last_layer) & (sub["module"] == "rc2")]["accuracy"].iloc[0],
                    "fixed": sub[(sub["layer"] == last_layer) & (sub["module"] == "act")]["accuracy"].iloc[0],
                    "best_rc2": sub[sub["module"] == "rc2"]["accuracy"].max(),
                    "best_act": sub[sub["module"] == "act"]["accuracy"].max(),
                }
            )
        evaluations = pd.DataFrame(rows)
        print(f"\n{scope} OOD vision evaluations: {len(evaluations)}")
        for strategy, label in [
            ("fixed", "Fixed ACT@100%"),
            ("best_rc2", "Best depth on RC2"),
            ("best_act", "Best depth on ACT"),
        ]:
            wins = (evaluations[strategy] > evaluations["standard"]).sum()
            gain = (evaluations[strategy] - evaluations["standard"]).mean()
            print(f"{label}: {wins}/{len(evaluations)} ({100 * wins / len(evaluations):.1f}%)")
            print(f"{label} average gain: {gain:.2f} pp")

    print_statistics(df, "All model sizes")
    large_df = df[df["model_size"] == df["model_name"].map(REPRESENTATIVE_VFMs)]
    print_statistics(large_df, "Large variants")


# ------------------------------------------------------------------------------
# LLMs plotting functions
# ------------------------------------------------------------------------------
def plot_model_family_language(filename: str, save: bool = False, ncol: int = 1) -> None:
    r"""
    Plot the gain over standard probing for each LLM family, one subplot
    per family, with grouped bars (layer/module/both) for each model
    size within that family.

    Three types of gains over standard probing (over the last-layer rc2
    output) are considered:
        - gain by optimizing over layers (rc2 fixed)
        - gain by optimizing over modules (last layer fixed)
        - gain by optimizing over layers and modules jointly

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    ncol: int, default=1
        Number of columns in the legend.
    """
    # Figure
    width = 5.5
    height = 4.4
    ncols = len(LLM_FAMILIES)
    figsize = ncols * width, height
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize, sharey=True)

    # Recover aggregate results.
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep language models
    pattern = "|".join(LLMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    gain_types = ["gain", "gain_module", "gain_layer"]
    gain_labels = ["oracle", "module", "layer"]
    model_labels = {
        "qwen3": ["Qwen3 1.7B", "Qwen3 4B", "Qwen3 8B"],
        "llama3": ["Llama3.2 1B", "Llama3.2 3B", "Llama3.1 8B"],
        "gemma3": ["Gemma3 1B", "Gemma3 4B", "Gemma3 12B"],
    }
    for ax, (model_family, model_order) in zip(axes, LLM_FAMILIES.items(), strict=True):
        sub = df[df["model_name"].isin(model_order)].set_index("model_name").reindex(model_order)
        agg_long = sub.reset_index().melt(
            id_vars="model_name",
            value_vars=gain_types,
            var_name="gain_type",
            value_name="gain_value",
        )
        agg_long["gain_type"] = agg_long["gain_type"].map(dict(zip(gain_types, gain_labels, strict=False)))
        sns.barplot(
            data=agg_long,
            x="model_name",
            y="gain_value",
            hue="gain_type",
            order=model_order,
            hue_order=gain_labels,
            palette=GAIN_PALETTE,
            ax=ax,
        )

        # Visualization
        ax.grid(axis="y", alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1)
        ax.legend().remove()
        ax.set_title(LLM_NAMES[model_family], fontsize=FONTSIZE)
        ax.set_xlabel("")
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels([model_name for model_name in model_labels[model_family]])
        if ax == axes[0]:
            ax.set_ylabel("Gain over standard probing (pp)", fontsize=FONTSIZE)
    sns.despine(fig, axes[0], trim=True, right=True, offset=10)

    # Common legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    labels = [LABEL[key] for key in gain_labels]
    legend = fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.94, 0.9),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=True,
        fontsize=LEG_FONTSIZE,
    )
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(10)
    plt.tight_layout()
    if save:
        figname = f"model_family_language_{filename}"
        save_plot(figname=figname)


def plot_probing_depth_language(filename: str, save: bool = False) -> None:
    r"""
    Evolution of the act and rc2 probing accuracy along with the depth.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 5.5
    height = 5
    ncols = len(REPRESENTATIVE_LLMs)
    figsize = ncols * width, height
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize)

    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep language models
    pattern = "|".join(LLMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    model_labels = {"qwen3": "Qwen3 1.7B", "llama3": "Llama3.2 1B", "gemma3": "Gemma3 1B"}
    tasks = df["task_name"].unique()
    for i, (model_family, model_name) in enumerate(REPRESENTATIVE_LLMs.items()):
        ax = axes[i]
        for module in ["rc2", "ln2"]:
            subset = df[(df["model_name"] == model_name) & (df["module"] == module)].copy()
            depth = []
            values = []
            for task_name in tasks:
                df_task = subset[subset.task_name == task_name]
                df_task = df_task.sort_values("layer")
                n_layers = df_task["layer"].max()
                if len(depth) == 0:
                    depth = df_task["layer"] / n_layers
                values.append(df_task["accuracy"])
            depth = 100 * depth
            acc = np.asarray(values).mean(axis=0)
            ax.plot(
                depth,
                acc,
                color=MODULE_PALETTE[module],
                linewidth=LINEWIDTH,
                linestyle="-" if module != "rc2" else "--",
                label=module.upper(),
            )

            # Mark the highest ln2 performance
            if module == "ln2":
                depth = np.asarray(depth)
                index = np.argmax(acc)
                ax.scatter(
                    depth[index],
                    acc[index],
                    label="Best LN2",
                    edgecolor="#333333",
                    s=300,
                    c=MODULE_PALETTE[module],
                    marker="*",
                    zorder=3,
                )

        # Visualization
        ax.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1, labelsize=FONTSIZE)
        if i == 0:
            yticks = [47, 53, 59]
        elif i == 1:
            yticks = [46, 53, 60]
        else:
            yticks = [47, 55, 63]
        ax.set_yticks(yticks)
        ax.set_xlabel(r"Layer depth ($\%$)", fontsize=FONTSIZE)
        if i == 0:
            ax.set_ylabel(r"Accuracy ($\%$)", fontsize=FONTSIZE)
        ax.set_title(model_labels[model_family], fontsize=FONTSIZE)
    sns.despine(fig, axes[0], trim=True, right=True, offset=10)

    # Common legend
    legend = axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.7, 0.45),
        fancybox=True,
        borderaxespad=0,
        shadow=False,
        frameon=True,
        handlelength=1.9,
        fontsize=LEG_FONTSIZE,
    )
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(10)
    plt.tight_layout()
    if save:
        figname = f"probing_depth_language_{filename}"
        save_plot(figname=figname)


def plot_probing_depth_all_language(filename: str, save: bool = False) -> None:
    r"""
    Evolution of the act and rc2 probing accuracy along with the depth.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 5.5
    height = 5
    nrows = 3
    ncols = 3
    figsize = ncols * width, nrows * height
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)

    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep language models
    pattern = "|".join(LLMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    model_labels = {
        "qwen3": ["Qwen3 1.7B", "Qwen3 4B", "Qwen3 8B"],
        "llama3": ["Llama3.2 1B", "Llama3.2 3B", "Llama3.1 8B"],
        "gemma3": ["Gemma3 1B", "Gemma3 4B", "Gemma3 12B"],
    }
    tasks = df["task_name"].unique()
    for i, (model_family, model_names) in enumerate(LLM_FAMILIES.items()):
        for j, model_name in enumerate(model_names):
            ax = axes[i, j]
            for module in ["rc2", "ln2"]:
                subset = df[(df["model_name"] == model_name) & (df["module"] == module)].copy()
                depth = []
                values = []
                for task_name in tasks:
                    df_task = subset[subset.task_name == task_name]
                    df_task = df_task.sort_values("layer")
                    n_layers = df_task["layer"].max()
                    if len(depth) == 0:
                        depth = df_task["layer"] / n_layers
                    values.append(df_task["accuracy"])
                depth = 100 * depth
                acc = np.asarray(values).mean(axis=0)
                ax.plot(
                    depth,
                    acc,
                    color=MODULE_PALETTE[module],
                    linewidth=LINEWIDTH,
                    linestyle="-" if module != "rc2" else "--",
                    label=module.upper(),
                )

                # Mark the highest ln2 performance
                if module == "ln2":
                    depth = np.asarray(depth)
                    index = np.argmax(acc)
                    ax.scatter(
                        depth[index],
                        acc[index],
                        label="Best LN2",
                        edgecolor="#333333",
                        s=300,
                        c=MODULE_PALETTE[module],
                        marker="*",
                        zorder=3,
                    )

            # Visualization
            ax.grid(alpha=ALPHA_GRID, lw=1.3)
            ax.spines["left"].set_linewidth(1)
            ax.spines["right"].set_linewidth(1)
            ax.spines["top"].set_linewidth(1)
            ax.spines["bottom"].set_linewidth(1)
            ax.tick_params(axis="both", direction="out", length=5, width=1, labelsize=FONTSIZE)
            if i == 0:
                if j == 0:
                    yticks = [47, 53, 59]
                elif j == 1:
                    yticks = [47, 54, 61]
                else:
                    yticks = [47, 55, 63]

            elif i == 1:
                if j == 0:
                    yticks = [46, 53, 60]
                elif j == 1:
                    yticks = [46, 54, 62]
                else:
                    yticks = [46, 55, 64]
            else:
                if j == 0:
                    yticks = [47, 55, 63]
                elif j == 1:
                    yticks = [47, 56, 65]
                else:
                    yticks = [47, 56, 65]
            ax.set_yticks(yticks)
            if i == nrows - 1:
                ax.set_xlabel(r"Layer depth ($\%$)", fontsize=FONTSIZE)
            if j == 0:
                ax.set_ylabel(r"Accuracy ($\%$)", fontsize=FONTSIZE)
            ax.set_title(model_labels[model_family][j], fontsize=FONTSIZE)
    sns.despine(fig, axes[0, 0], trim=True, right=True, offset=10)

    # Common legend
    legend = axes[0, 0].legend(
        loc="upper center",
        bbox_to_anchor=(0.7, 0.45),
        fancybox=True,
        borderaxespad=0,
        shadow=False,
        frameon=True,
        handlelength=1.9,
        fontsize=LEG_FONTSIZE,
    )
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(10)
    plt.tight_layout()
    if save:
        figname = f"probing_depth_all_language_{filename}"
        save_plot(figname=figname)


def plot_oracle_recovery_language(filename: str, save: bool = False) -> None:
    r"""
    Recover the table of oracle gain recovery (computed as gain / oracle_gain) averaged over all datasets and models.

    Strategies:
        - Best layer on rc2
        - Fixed (fixed 60% of depth on ln2)
        - Best layer on ln2

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 5
    height = 4
    figsize = width, height
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()
    df = df.groupby(["model_name", "layer", "module"])["accuracy"].mean().reset_index()

    # Keep LLM models
    all_models = [m for models in LLM_FAMILIES.values() for m in models]
    df = df[df["model_name"].isin(all_models)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    order = ["Best RC2", "Fixed", "Best LN2"]
    groups = "model_name"
    rows = []
    for key, sub in df.groupby(groups):
        last_layer = sub["layer"].max()

        # Standard
        baseline = sub[(sub.layer == last_layer) & (sub.module == "rc2")].accuracy.iloc[0]

        # Best RC2
        rc2_best = sub[sub.module == "rc2"].accuracy.max()

        # Fixed: 60% of depth on ln2
        heuristic_depth = 0.6
        heuristic_layer = int(heuristic_depth * last_layer)
        heuristic = sub[(sub.layer == heuristic_layer) & (sub.module == "ln2")].accuracy.iloc[0]

        # Best ln2
        ln2_best = sub[sub.module == "ln2"].accuracy.max()

        # Recover strategies
        strategies = {
            "Best RC2": rc2_best,
            "Fixed": heuristic,
            "Best LN2": ln2_best,
        }

        # Recover oracle recovery
        oracle = sub.accuracy.max()
        oracle_gain = oracle - baseline

        # Ignore zero oracle gains
        if oracle_gain > 0:
            for strategy, acc in strategies.items():
                # If a strategy does not provide any improvement, we assume it recovers 0% of the oracle gain
                gain = np.max((acc - baseline, 0))
                recovery = gain / oracle_gain if gain > 0 else 0
                rows.append(
                    {
                        "model_name": key,
                        "strategy": strategy,
                        "recovery": recovery * 100,
                    }
                )
    summary = pd.DataFrame(rows)
    means = summary.groupby("strategy")["recovery"].mean().reindex(order)
    print(means)
    sns.barplot(
        data=summary,
        x="strategy",
        y="recovery",
        order=order,
        estimator=np.mean,
        errorbar=("ci", 95),
        palette=[MODULE_PALETTE["rc2"], MODULE_PALETTE["ln2"], MODULE_PALETTE["ln2"]],
        legend=False,
        err_kws={
            "color": "#333333",
            "linewidth": 0.5,
            "linestyle": "--",
        },
        capsize=0.15,
        ax=ax,
    )
    heuristic_idx = order.index("Fixed")
    bar = ax.patches[heuristic_idx]
    bar.set_hatch("//")

    # Visualization
    ax.grid(axis="y", alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xticklabels(["Best RC2", r"$\mathtt{LN2@60}$", "Best LN2"])
    ax.set_ylim(20, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xlabel("")
    ax.set_ylabel(r"Oracle gain recovered ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        save_plot(figname=f"oracle_recovery_language_{filename}")


def plot_winning_distribution_language(filename: str, save: bool = False) -> None:
    r"""
    Plot the distribution of winning layer-module locations for language models.

    Each model contributes one vote, split equally across tied winners. The
    layer-depth axis uses the same orientation and geometry as the vision plot.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Recover and aggregate results over language-model tasks.
    df = pd.read_csv(RESULT_DIR / f"{filename}.csv")
    pattern = "|".join(LLMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    df = df.groupby(["model_name", "layer", "module"], as_index=False, observed=True)["accuracy"].mean()
    group_cols = ["model_name"]
    grouped = df.groupby(group_cols, observed=True)
    last_layer = grouped["layer"].transform("max")
    oracle_accuracy = grouped["accuracy"].transform("max")
    winners = df.loc[np.isclose(df["accuracy"], oracle_accuracy, atol=1e-8, rtol=0)].copy()
    winners["depth_pct"] = 100 * winners["layer"] / last_layer.loc[winners.index].clip(lower=1)
    n_winners = winners.groupby(group_cols, observed=True)["module"].transform("size")
    winners["weight"] = 1 / n_winners
    n_units = int(grouped.ngroups)

    # Bin depth to make the winning mass easy to read.
    depth_bin = 5
    depth_edges = np.arange(0, 100 + depth_bin, depth_bin, dtype=float)
    depth_left = depth_edges[:-1]
    bin_index = np.clip(
        np.digitize(winners["depth_pct"], depth_edges, right=False) - 1,
        0,
        len(depth_left) - 1,
    )
    module_index = winners["module"].map({module: i for i, module in enumerate(MODULE_ORDER)}).to_numpy()
    grid = np.zeros((len(MODULE_ORDER), len(depth_left)), dtype=float)
    np.add.at(grid, (module_index, bin_index), 100 * winners["weight"].to_numpy() / n_units)

    # Draw one consistent color per module; height, not color, encodes frequency.
    depth_width = depth_bin * 0.78
    module_width = 0.38
    for module_idx, module in enumerate(MODULE_ORDER):
        columns = np.flatnonzero(grid[module_idx] > 0)
        if len(columns) == 0:
            continue
        alpha = 1.0 if module == "act" else 0.72
        ax.bar3d(
            np.full(len(columns), module_idx - module_width / 2),
            depth_left[columns],
            np.zeros(len(columns)),
            np.full(len(columns), module_width),
            np.full(len(columns), depth_width),
            grid[module_idx, columns],
            color=MODULE_PALETTE[module],
            alpha=alpha,
            edgecolor="white",
            linewidth=0.25,
            shade=True,
        )

    # Mark the single most frequent winning location.
    best_module_idx, best_depth_idx = np.unravel_index(np.argmax(grid), grid.shape)
    best_value = grid[best_module_idx, best_depth_idx]
    if best_value > 0:
        ax.scatter(
            best_module_idx,
            depth_left[best_depth_idx] + depth_width / 2,
            best_value + 1.5,
            marker="*",
            s=320,
            color="#d62728",
            edgecolor="white",
            linewidth=0.8,
            depthshade=False,
            zorder=10,
        )

    # Visualization
    module_labels = [module.upper() if module != "act" else "Act" for module in MODULE_ORDER]
    z_tick_max = max(10, 10 * int(np.ceil(grid.max() / 10)))
    z_limit = z_tick_max * 1.1
    ax.set(
        xlabel="\nModule",
        ylabel="\nLayer depth (%)",
        zlabel="\nWinning frequency (%)",
        xlim=(-0.6, len(MODULE_ORDER) - 0.4),
        ylim=(0, 100),
        zlim=(0, z_limit),
        xticks=np.arange(len(MODULE_ORDER)),
        xticklabels=module_labels,
        yticks=[0, 25, 50, 75, 100],
    )
    ax.set_zticks(np.arange(0, z_tick_max + 1, 10))
    ax.view_init(elev=25, azim=40)
    ax.set_box_aspect((2.5, 1.6, 1.8))
    ax.tick_params(axis="both", direction="out", length=5, width=1, labelsize=FONTSIZE - 3, pad=6)
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="z", labelsize=FONTSIZE - 3, pad=4)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.96, 0.96, 0.96, 1.0))
        axis.pane.set_edgecolor("white")
    ax.set_title("Winning layer–module locations", fontsize=FONTSIZE, pad=18)
    fig.set_layout_engine("none")

    if save:
        save_plot(figname=f"winning_distribution_language_{filename}")


# ------------------------------------------------------------------------------
# LLMs table functions
# ------------------------------------------------------------------------------
def table_loto_depth_language(filename: str) -> None:
    r"""
    Leave-one-task-out selection of the fixed language probing depth.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    all_models = [m for models in LLM_FAMILIES.values() for m in models]
    df = df[df["model_name"].isin(all_models)].copy()

    # LOTO
    grid_step = 5
    tasks = df["task_name"].unique()
    candidate_depths = np.arange(0, 101, grid_step)
    last_layers = df.groupby("model_name")["layer"].max()
    rows = []
    for held_out in tasks:
        train = df[df["task_name"] != held_out]
        scores = []
        for depth in candidate_depths:
            task_scores = []
            for _, task in train[train["module"] == "ln2"].groupby("task_name"):
                model_scores = []
                for model_name, model in task.groupby("model_name"):
                    layer = int(depth / 100 * last_layers[model_name])
                    model_scores.append(model[model["layer"] == layer]["accuracy"].iloc[0])
                task_scores.append(np.mean(model_scores))
            scores.append(np.mean(task_scores))
        train_score, selected_depth = max(
            zip(scores, candidate_depths, strict=False),
            key=lambda item: (item[0], item[1]),
        )

        held_out_scores = []
        for model_name, model in df[(df["task_name"] == held_out) & (df["module"] == "ln2")].groupby("model_name"):
            layer = int(selected_depth / 100 * last_layers[model_name])
            held_out_scores.append(model[model["layer"] == layer]["accuracy"].iloc[0])
        rows.append(
            {
                "task": held_out,
                "selected_depth": selected_depth,
                "train_score": train_score,
                "held_out_accuracy": np.mean(held_out_scores),
            }
        )

    table = pd.DataFrame(rows).sort_values("task")
    choice = table["selected_depth"].mode().iloc[0]
    print(f"LOTO selected depth: {choice}%")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


def table_performance_comparison_language(filename: str) -> None:
    r"""Performance comparison between standard probing, heuristic probing and oracle probing."""
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()
    df = df.groupby(["model_name", "layer", "module"])["accuracy"].mean().reset_index()

    # Keep LLM models
    all_models = [m for models in LLM_FAMILIES.values() for m in models]
    df = df[df["model_name"].isin(all_models)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    rows = []
    groups = "model_name"
    for model_name, sub in df.groupby(groups):
        last_layer = sub.layer.max()
        standard = sub[(sub.layer == last_layer) & (sub.module == "rc2")].accuracy.iloc[0]
        heuristic_depth = 0.6
        heuristic_layer = int(heuristic_depth * last_layer)
        heuristic = sub[(sub.layer == heuristic_layer) & (sub.module == "ln2")].accuracy.iloc[0]
        oracle = sub.accuracy.max()
        rows.extend(
            [
                {
                    "model_name": model_name,
                    "strategy": "Standard",
                    "accuracy": standard,
                },
                {
                    "model_name": model_name,
                    "strategy": "Fixed",
                    "accuracy": heuristic,
                },
                {
                    "model_name": model_name,
                    "strategy": "Oracle",
                    "accuracy": oracle,
                },
            ]
        )
    summary = pd.DataFrame(rows)
    table = summary.pivot(index="strategy", columns="model_name", values="accuracy")
    table = table.loc[["Standard", "Fixed", "Oracle"]]
    print(table.to_latex(float_format="%.1f", bold_rows=True, escape=False))


def table_global_statistics_language(filename: str) -> None:
    r"""
    Compute global statistics for fixed and oracle language strategies.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    """
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep LLM models
    all_models = [m for models in LLM_FAMILIES.values() for m in models]
    df = df[df["model_name"].isin(all_models)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    heuristic_depth = 0.6
    last_layers = df.groupby("model_name")["layer"].max()
    rows = []
    for (task_name, model_name), sub in df.groupby(["task_name", "model_name"]):
        last_layer = last_layers[model_name]
        standard = sub[(sub["layer"] == last_layer) & (sub["module"] == "rc2")]["accuracy"].iloc[0]
        heuristic_layer = int(heuristic_depth * last_layer)
        fixed = sub[(sub["layer"] == heuristic_layer) & (sub["module"] == "ln2")]["accuracy"].iloc[0]
        best_rc2 = sub[sub["module"] == "rc2"]["accuracy"].max()
        best_ln2 = sub[sub["module"] == "ln2"]["accuracy"].max()
        rows.append(
            {
                "task": task_name,
                "model_name": model_name,
                "standard": standard,
                "fixed": fixed,
                "best_rc2": best_rc2,
                "best_ln2": best_ln2,
            }
        )

    evaluations = pd.DataFrame(rows)
    tasks = evaluations.groupby("task")[["standard", "fixed", "best_rc2", "best_ln2"]].mean()

    def print_statistics(strategy: str, label: str) -> None:
        evaluation_wins = (evaluations[strategy] > evaluations["standard"]).sum()
        task_wins = (tasks[strategy] > tasks["standard"]).sum()
        evaluation_gain = (evaluations[strategy] - evaluations["standard"]).mean()
        task_gain = (tasks[strategy] - tasks["standard"]).mean()
        print(
            f"{label} model-task evaluations where strategy improves: "
            f"{evaluation_wins}/{len(evaluations)} ({100 * evaluation_wins / len(evaluations):.1f}%)"
        )
        print(
            f"{label} tasks where strategy improves on average: "
            f"{task_wins}/{len(tasks)} ({100 * task_wins / len(tasks):.1f}%)"
        )
        print(f"{label} average gain over model-task evaluations: {evaluation_gain:.2f} pp")
        print(f"{label} average gain over tasks: {task_gain:.2f} pp")

    print("\nFixed module comparison: LN2@60% vs. standard RC2@100%.")
    print_statistics("fixed", "Fixed LN2")
    print("\nDepth-optimized module comparison.")
    print_statistics("best_rc2", "Depth-RC2")
    print_statistics("best_ln2", "Depth-LN2")


# ------------------------------------------------------------------------------
# Results functions
# ------------------------------------------------------------------------------
def plot_figures() -> None:
    r"""VFMs plots."""
    # # CDF
    # save = True
    # plot_cdf_vision(filename="cls", save=save)

    # # Dataset difficulty
    # save = True
    # filename = "cls"
    # plot_transfer_difficulty_vision(filename=filename, save=save)

    # # Cifar10-C
    # save = True
    # filename = "cifar10_c"
    # plot_cifar10c_severity(filename=filename, save=save)
    # plot_cifar10c_optimal_depth(filename=filename, save=save)

    # # Pretraining scale
    # save = True
    # filename = "cls"
    # plot_pretraining_scale_vision(filename=filename, save=save)

    # # Model family
    # save = True
    # filename = "cls"
    # plot_model_family_vision(filename=filename, save=save)

    # # Probing depth
    # save = True
    # filename = "cls"
    # plot_probing_depth_vision(filename=filename, save=save)
    # for model_name in ["vit", "dinov2", "dinov3", "aimv2"]:
    #     plot_probing_depth_family_vision(filename=filename, model_name=model_name, save=save)

    # Oracle winning distribution
    save = True
    filename = "cls"
    plot_heatmap_winning_frequency_vision(filename=filename, save=save)

    # # Oracle recovery
    # save = True
    # filename = "cls"
    # plot_oracle_recovery_vision(filename=filename, save=save)

    r"""LLMs plots."""
    # # Model family
    # save = True
    # filename = "mteb"
    # plot_model_family_language(filename=filename, save=save)

    # # Probing depth
    # save = True
    # filename = "mteb"
    # plot_probing_depth_language(filename=filename, save=save)
    # plot_probing_depth_all_language(filename=filename, save=save)

    # # Oracle winning distribution
    # save = True
    # filename = "mteb"
    # plot_heatmap_winning_frequency_language(filename=filename, save=save)

    # # Oracle recovery
    # save = True
    # filename = "mteb"
    # plot_oracle_recovery_language(filename=filename, save=save)

    # # Oracle winning distribution
    # save = True
    # filename = "mteb"
    # plot_winning_distribution_language(filename=filename, save=save)


def get_tables() -> None:
    r"""Number of runs"""
    # table_runs()

    r"""VFMs tables."""
    # # Model size
    # filename = "cls"
    # table_model_size_vision(filename=filename)

    # # Winning module frequency
    # filename = "cls"
    # table_winning_module_frequency_vision(filename=filename)

    # # LODO depth selection
    # filename = "cls"
    # table_lodo_depth_vision(filename=filename)

    # # Performance comparisons
    # filename = "cls"
    # table_performance_comparison_vision(filename=filename)
    # table_performance_comparison_all_vision(filename=filename)

    # # Global statistics
    # filename = "cls"
    # table_global_statistics_vision(filename=filename)
    # table_global_statistics_ood_vision(filename=filename)

    r"""LLMs tables."""
    # # LOTO depth selection
    # filename = "mteb"
    # table_loto_depth_language(filename=filename)

    # # Performance comparisons
    # filename = "mteb"
    # table_performance_comparison_language(filename=filename)

    # # Global statistics
    # filename = "mteb"
    # table_global_statistics_language(filename=filename)


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(
        {
            "plot": plot_figures,
            "table": get_tables,
        }
    )


# %% CLI
if __name__ == "__main__":
    main()
# %%
