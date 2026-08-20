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
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare, spearmanr

from waldo.infra import FIGURE_DIR, RESULT_DIR

logger = logging.getLogger("waldo.analysis")

# Variables
MODULE_ORDER = ["ln1", "mha", "rc1", "ln2", "fc1", "act", "fc2", "rc2"]
LABEL = {
    "oracle": r"$\mathscr{G}_\mathrm{oracle}$",
    "layer": r"$\mathscr{G}_\mathrm{layer}$",
    "module": r"$\mathscr{G}_\mathrm{module}$",
}

# VFMs
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
REPRESENTATIVE_VFMs = {"vit": "large", "dinov2": "large", "aimv2": "large", "dinov3": "vitl16"}
VFM_NAMES = {"vit": "ViT", "dinov2": "DINOv2", "aimv2": "AIMv2", "dinov3": "DINOv3"}
VFM_SIZES = {
    "vit": ["base", "large", "huge"],
    "dinov2": ["small", "base", "large", "giant"],
    "aimv2": ["large", "huge", "1B", "3B"],
    "dinov3": ["vits16", "vitb16", "vitl16", "vit7b16"],
}
REPRESENTATIVE_DATASETS = ["aircraft", "domainnet-sketch", "flowers102"]
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
    "qwen3": {"Qwen3-1.7B": 1.7, "Qwen3-4B": 4, "Qwen3-8B": 8},
    "llama3": {"Llama-3.2-1B": 1, "Llama-3.2-3B": 3, "Llama-3.1-8B": 8},
    "gemma3": {"gemma-3-1b-pt": 1, "gemma-3-4b-pt": 4, "gemma-3-12b-pt": 12},
}

# Plot aesthetic
FONTSIZE = 19
LEG_FONTSIZE = 19
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
# VFMs plotting functions
# ------------------------------------------------------------------------------
def plot_cdf_vision(filename: str, ncol: int = 3, save: bool = False) -> None:
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
    ncol: int, default=3
        Number of legend columns.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 6
    height = 5
    figsize = width, height
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    # Zoomed window (x and y axis are padded to avoid overlapping splines)
    ZOOM_XLIM = (-0.05, 0.45)
    ZOOM_YLIM = (70, 100)
    axins = ax.inset_axes((0.35, 0.38, 0.24, 0.44))

    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    for col, label in [("gain_layer", "layer"), ("gain_module", "module"), ("gain", "oracle")]:
        gains = np.sort(df[col].to_numpy())
        percentage = 100 * (1 - np.arange(1, len(gains) + 1) / len(gains))
        ax.plot(gains, percentage, linewidth=LINEWIDTH, color=GAIN_PALETTE[label], label=LABEL[label])
        axins.plot(gains, percentage, linewidth=LINEWIDTH, color=GAIN_PALETTE[label])

    # Subplot limits
    axins.set_xlim(*ZOOM_XLIM)
    axins.set_ylim(*ZOOM_YLIM)
    yticks = [70, 80, 90, 100]
    axins.set_yticks(yticks)
    axins.grid(alpha=ALPHA_GRID, lw=1.0)
    axins.tick_params(axis="both", direction="out", length=3, width=1, labelsize=FONTSIZE * ZOOM_SCALE)
    for spine in axins.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("0.6")

    # Align zoomed window and subplot
    ax.indicate_inset_zoom(axins, edgecolor="0.15", linewidth=1.0, linestyle="--")

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
    ax.set_ylabel(r"Percentage of runs with gain $> x$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.05),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=False,
        fontsize=LEG_FONTSIZE,
    )
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
    # Figure
    width = 6
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

    # Dataset difficulty
    summary = (
        df.groupby("dataset_name")
        .agg(max_accuracy=("baseline_accuracy", "max"), mean_gain=("gain", "mean"))
        .reset_index()
    )
    summary["best_error"] = 100 - summary["max_accuracy"]

    # Plotting
    ax.scatter(
        summary["best_error"],
        summary["mean_gain"],
        s=50,
        zorder=3,
        color="#a291e1",
        edgecolors="#333333",
        linewidths=MARKER_LINEWIDTH,
    )

    # Regression
    rho, pval = spearmanr(summary["max_accuracy"], summary["mean_gain"])
    sns.regplot(
        data=summary,
        x="best_error",
        y="mean_gain",
        color="#a291e1",
        scatter=False,
        ci=None,
        line_kws={"linewidth": 2, "alpha": 0.8},
        ax=ax,
    )
    handle = Line2D(
        [0], [0], color="#a291e1", linewidth=2, label=rf"$\rho= {rho:.2f}$" + "\n" + rf"$\text{{p-value}}= {pval:.3f}$"
    )
    legend = ax.legend(
        handles=[handle],
        loc="upper center",
        bbox_to_anchor=(0.275, 0.9),
        fancybox=True,
        borderaxespad=0,
        shadow=False,
        frameon=True,
        handlelength=1.9,
        fontsize=12,
    )
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(10)

    # Annotate
    summary_subset = summary.copy()
    summary_subset = summary_subset[summary_subset["dataset_name"].isin(REPRESENTATIVE_DATASETS)]
    for _, row in summary_subset.iterrows():
        ax.annotate(
            DATASETS[row["dataset_name"]],
            (row["best_error"], row["mean_gain"]),
            xytext=(8, 0),
            textcoords="offset points",
            color="#333333",
            fontsize=15,
            va="center",
            ha="left",
        )

    # Visualization
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    xticks = [0, 6, 12, 18, 24]
    ax.set_xticks(xticks)
    yticks = [0, 2, 4, 6]
    ax.set_yticks(yticks)
    ax.set_xlabel(r"Lowest achievable error ($\%$)", fontsize=FONTSIZE)
    ax.set_ylabel(r"Oracle gain $\mathscr{G}_\mathrm{oracle}$ (pp)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        figname = f"dataset_difficulty_vision_{filename}"
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
    width = 6
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
        palette=["#a291e1", "#a291e1", "#a291e1"],
        legend=False,
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
    yticks = [0, 1, 2]
    ax.set_yticks(yticks)
    ax.set_xlabel("Corruption severity", fontsize=FONTSIZE)
    ax.set_ylabel(r"Oracle gain $\mathscr{G}_\mathrm{oracle}$ (pp)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        save_plot(figname=f"cifar10c_severity_{filename}")


def plot_pretraining_scale_vision(filename: str, ncol: int = 3, save: bool = False) -> None:
    r"""
    Plot the average gains over standard probing for different pretraining scales.

    We follow the scaling law literature for LLMs [1, 2]_ and consider the total amount of tokens seen during
    pretraining. Notably, this takes into considerations multiple epochs, i.e., the number of image exposure, and
    the number of tokens in each image. This depends on the patch size and image resolution. For simplicity (and
    since this does not impact the order of magnitude), we only consider visual tokens. In particular, we do not
    account for CLS or register tokens. In summary, we have: image exposure x token per image.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    ncol: int, default=3
        Number of columns for the legend.
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
    width = 6
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

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_VFMs.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = df[mask]

    # Pretraining scale
    pretraining_tokens = {"vit": 2.5e11, "dinov2": 1.80e12, "aimv2": 3.15e12, "dinov3": 3.91e12}
    df["pretraining_tokens"] = df["model_name"].map(pretraining_tokens)

    # Results
    stats = {}
    models = ["vit", "dinov2", "aimv2", "dinov3"]
    stats["layer"] = [np.mean(df[df["model_name"] == model]["gain_layer"].to_numpy()) for model in models]
    stats["module"] = [np.mean(df[df["model_name"] == model]["gain_module"].to_numpy()) for model in models]
    stats["oracle"] = [np.mean(df[df["model_name"] == model]["gain"].to_numpy()) for model in models]
    scales = list(pretraining_tokens.values())
    for key, values in stats.items():
        ax.plot(scales, values, linewidth=LINEWIDTH, color=GAIN_PALETTE[key], label=LABEL[key])
        for model_name, x, y in zip(models, scales, values, strict=True):
            ax.scatter(
                x,
                y,
                s=VFM_MARKER_SIZE[model_name],
                color=GAIN_PALETTE[key],
                edgecolor="#333333",
                linewidth=MARKER_LINEWIDTH,
                marker=VFM_MARKERS[model_name],
                zorder=2,
            )

    # Visualization
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    xticks = [2.5e11, 1.5e12, 2.75e12, 4e12]
    ax.set_xticks(xticks)
    ax.set_xticklabels([r"$250$B", r"$1.5$T", r"$2.75$T", r"$4$T"])
    yticks = [0.5, 2.5, 4.5]
    ax.set_yticks(yticks)
    ax.set_xlabel("Number of pretraining tokens", fontsize=FONTSIZE)
    ax.set_ylabel("Gain over standard probing (pp)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Marker legend
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="",
            marker=m,
            markerfacecolor="#e0e0e0",
            markeredgecolor="#333333",
            markersize=10 if "aim" not in model else 14,
            label=VFM_NAMES[model],
        )
        for model, m in VFM_MARKERS.items()
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.82, 0.9),
        fancybox=True,
        borderaxespad=0,
        shadow=False,
        frameon=True,
        handlelength=1.9,
        fontsize=12,
    )
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(10)
    ax.add_artist(legend)

    # Common legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.05),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=False,
        fontsize=LEG_FONTSIZE,
    )
    plt.tight_layout()
    if save:
        figname = f"pretraining_scale_vision_{filename}"
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
    # Figure
    width = 6
    height = 5
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
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_VFMs.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = df[mask]

    # Results
    model_order = list(REPRESENTATIVE_VFMs.keys())
    gain_types = ["gain_layer", "gain_module", "gain"]
    gain_labels = ["layer", "module", "oracle"]
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
    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.08),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=False,
        fontsize=22,
    )
    plt.tight_layout()
    if save:
        figname = f"model_family_vision_{filename}"
        save_plot(figname=figname)


def plot_probing_depth_vision(filename: str, ncol: int = 2, save: bool = False) -> None:
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
    width = 6
    height = 5
    ncols = len(REPRESENTATIVE_VFMs)
    figsize = ncols * width, height
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize)

    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_VFMs.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = df[mask]

    # Results
    dataset_name = "domainnet-sketch"
    for i, model_name in enumerate(REPRESENTATIVE_VFMs):
        ax = axes[i]
        model_size = REPRESENTATIVE_VFMs[model_name]
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
        yticks = [5, 30, 55, 80] if i > 0 else [10, 25, 40, 55]
        ax.set_yticks(yticks)
        ax.set_xlabel(r"Layer depth ($\%$)", fontsize=FONTSIZE)
        if i == 0:
            ax.set_ylabel(r"Accuracy ($\%$)", fontsize=FONTSIZE)
        ax.set_title(VFM_NAMES[model_name], fontsize=FONTSIZE)
    sns.despine(fig, axes[0], trim=True, right=True, offset=10)

    # Common legend
    legend = axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.75, 0.3),
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
        figname = f"probing_depth_vision_{filename}"
        save_plot(figname=figname)


def plot_probing_depth_all_vision(filename: str, save: bool = False) -> None:
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
    width = 6
    height = 5
    nrows = len(REPRESENTATIVE_DATASETS)
    ncols = len(REPRESENTATIVE_VFMs)
    figsize = ncols * width, nrows * height
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)

    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_VFMs.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = df[mask]

    # Results
    for i, dataset_name in enumerate(REPRESENTATIVE_DATASETS):
        for j, model_name in enumerate(REPRESENTATIVE_VFMs):
            ax = axes[i, j]
            model_size = REPRESENTATIVE_VFMs[model_name]
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
                    yticks = [10, 23, 36, 49]
                elif j == 1:
                    yticks = [3, 24, 46, 67]
                elif j == 2:
                    yticks = [5, 29, 53, 77]
                else:
                    yticks = [3, 28, 53, 81]
            elif i == 1:
                yticks = [5, 30, 55, 80] if j > 0 else [10, 25, 40, 55]
            else:
                if j == 0:
                    yticks = [40, 58, 76, 94]
                elif j == 1:
                    yticks = [10, 39, 68, 97]
                elif j == 2:
                    yticks = [34, 56, 78, 100]
                else:
                    yticks = [25, 50, 75, 100]
            ax.set_yticks(yticks)
            if i == nrows - 1:
                ax.set_xlabel(r"Layer depth ($\%$)", fontsize=FONTSIZE)
            if j == 0:
                ax.set_ylabel(r"Accuracy ($\%$)", fontsize=FONTSIZE)
            ax.set_title(VFM_NAMES[model_name], fontsize=FONTSIZE)
    sns.despine(fig, axes[0, 0], trim=True, right=True, offset=10)

    # Common legend
    legend = axes[0, 0].legend(
        loc="upper center",
        bbox_to_anchor=(0.75, 0.3),
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
        figname = f"probing_depth_all_vision_{filename}"
        save_plot(figname=figname)


def plot_oracle_recovery_vision(filename: str, save: bool = False) -> None:
    r"""
    Recover the table of oracle gain recovery (computed as gain / oracle_gain) averaged over all datasets and models.

    Strategies:
        - Best layer on rc2
        - Heuristic (last layer on act)
        - Best layer on act

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 6
    height = 5
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

    # Results
    order = ["Best RC2", "Heuristic", "Best ACT"]
    groups = ["dataset_name", "model_name", "model_size"]
    rows = []
    for key, sub in df.groupby(groups):
        last_layer = sub["layer"].max()

        # Standard
        baseline = sub[(sub.layer == last_layer) & (sub.module == "rc2")].accuracy.iloc[0]

        # Best RC2, heuristic, best act
        rc2_best = sub[sub.module == "rc2"].accuracy.max()
        heuristic = sub[(sub.layer == last_layer) & (sub.module == "act")].accuracy.iloc[0]
        act_best = sub[sub.module == "act"].accuracy.max()
        strategies = {
            "Best RC2": rc2_best,
            "Heuristic": heuristic,
            "Best ACT": act_best,
        }

        # Recover oracle recovery
        oracle = sub.accuracy.max()
        oracle_gain = oracle - baseline

        # Ignore negligible oracle gains, i..e, less that 0.5 accuracy point
        if oracle_gain < 0.5:
            continue

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
    sns.barplot(
        data=summary,
        x="strategy",
        y="recovery",
        order=order,
        estimator=np.mean,
        errorbar=("ci", 95),
        palette=[MODULE_PALETTE["rc2"], MODULE_PALETTE["act"], MODULE_PALETTE["act"]],
        legend=False,
        err_kws={"color": "#333333", "linewidth": 1},
        capsize=0.15,
        ax=ax,
    )
    heuristic_idx = order.index("Heuristic")
    bar = ax.patches[heuristic_idx]
    bar.set_hatch("//")

    # Visualization
    ax.grid(axis="y", alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xticklabels(["Best RC2", "Heuristic", "Best Act"])
    ax.set_xlabel("")
    ax.set_ylabel(r"Oracle recovery ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        save_plot(figname=f"oracle_recovery_vision_{filename}")


# ------------------------------------------------------------------------------
# VFMs table functions
# ------------------------------------------------------------------------------
def table_model_size_vision(filename: str) -> None:
    r"""
    Friedman test to determine whether the model size has a significant impact on the gain
    per model family. The null hypothesis is that model size does not influence the gain.
    """
    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

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
        W = stat / (k * (n - 1))  # = chi2 / (k*(n-1)) for Friedman
        results.append(
            {
                "Family": family,
                "n_sizes": k,
                "n_datasets": n,
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
    """
    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep only vision foundation models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Results
    freq = df["best_module"].value_counts(normalize=True).mul(100)
    freq = freq.reindex(MODULE_ORDER, fill_value=0.0)

    table = freq.to_frame().T
    table.index = ["Winning freq. (%)"]
    print(table.to_latex(index=True, float_format="%.1f"))


def table_performance_comparison_vision(filename: str) -> None:
    r"""Performance comparison between standard probing, heuristic probing and oracle probing."""
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

    # Add the family level
    overall_rows = pd.concat({"Overall": overall_rows}, names=["family"])
    table = pd.concat([table, overall_rows])

    # Latex table
    print(table.to_latex(float_format="%.1f", multirow=True, bold_rows=True))


def table_performance_comparison_all_vision(filename: str) -> None:
    r"""Performance comparison across all model sizes."""
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

    # Add the family/size level (Overall has no size, so use a placeholder)
    overall_rows = pd.concat({("Overall", ""): overall_rows}, names=["family", "size"])
    table = pd.concat([table, overall_rows])

    # Latex table
    print(table.to_latex(float_format="%.1f", multirow=True, bold_rows=True))


# ------------------------------------------------------------------------------
# LLMs plotting functions
# ------------------------------------------------------------------------------
def plot_cdf_language(filename: str, ncol: int = 3, save: bool = False) -> None:
    r"""
    Plot the percentage of runs over LLMs for which the gain is strictly above a certain threshold.

    Three types of gains over standard probing (over the output of the last transformer block) are considered:
        - gain by optimizing over layers (probing done on the output of transformer blocks)
        - gain by optimizing over modules (probing done over modules of the last transformer block)
        - gain by optimizing over layers and modules (probing done over modules inside transformer blocks)

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    ncol: int, default=3
        Number of legend columns.
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 6
    height = 5
    figsize = width, height
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(LLMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
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
    ax.set_xlim(-0.1, 25)
    ax.set_ylim(-1, 101)
    yticks = [0, 25, 50, 75, 100]
    ax.set_yticks(yticks)
    ax.set_xlabel("Gain over standard probing (pp)", fontsize=FONTSIZE)
    ax.set_ylabel(r"Percentage of runs with gain $> x$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.05),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=False,
        fontsize=LEG_FONTSIZE,
    )
    plt.tight_layout()
    if save:
        figname = f"cdf_language_{filename}"
        save_plot(figname=figname)


def plot_model_family_language(filename: str, save: bool = False, ncol: int = 3) -> None:
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
    ncol: int, default=3
        Number of columns in the legend.
    """
    # Figure
    width = 6
    height = 5
    ncols = len(LLM_FAMILIES)
    figsize = ncols * width, height
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize, sharey=True)

    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()
    logger.info(f"Results are presented over a total of {len(df)} models.")

    # Results
    gain_types = ["gain_layer", "gain_module", "gain"]
    gain_labels = ["layer", "module", "oracle"]
    model_sizes = {
        "qwen3": ["Qwen3 1.7B", "Qwen3 4B", "Qwen3 8B"],
        "llama3": ["Llama3.2 1B", "Llama3.2 3B", "Llama3.1 8B"],
        "gemma3": ["Gemma3 1B", "Gemma3 4B", "Gemma3 12B"],
    }
    for ax, (family_name, model_order) in zip(axes, LLM_FAMILIES.items(), strict=True):
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
        ax.set_title(LLM_NAMES[family_name], fontsize=FONTSIZE)
        ax.set_xlabel("")
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels([model_name for model_name in model_sizes[family_name]])
        if ax == axes[0]:
            ax.set_ylabel("Gain over standard probing (pp)", fontsize=FONTSIZE)
    sns.despine(fig, axes[0], trim=True, right=True, offset=10)

    # Common legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    labels = [LABEL[key] for key in gain_labels]
    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.08),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=False,
        fontsize=22,
    )
    plt.tight_layout()
    if save:
        figname = f"model_family_language_{filename}"
        save_plot(figname=figname)


def plot_probing_depth_language(filename: str, save: bool = False) -> None:
    r"""Evolution of the act and rc2 probing accuracy along with the depth."""
    # Figure
    width = 6
    height = 5
    ncols = len(REPRESENTATIVE_LLMs)
    figsize = ncols * width, height
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize)

    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Results
    tasks = df["task_name"].unique()
    for i, (family_name, model_name) in enumerate(REPRESENTATIVE_LLMs.items()):
        ax = axes[i]
        for module in ["ln2", "rc2"]:
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

            # Mark the highest fc1 performance
            if module == "ln2":
                depth = np.asarray(depth)
                index = np.argmax(acc)
                ax.scatter(
                    depth[index],
                    acc[index],
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
        ax.set_title(LLM_NAMES[family_name], fontsize=FONTSIZE)
    sns.despine(fig, axes[0], trim=True, right=True, offset=10)

    # Common legend
    legend = axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.75, 0.3),
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
    r"""Evolution of the act and rc2 probing accuracy along with the depth over all LLMs."""
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Figure
    nrows = 3
    ncols = 4
    unit_width = 2  # each subplot spans 2 grid units
    figsize = ncols * 2 * WIDTH, nrows * 2 * HEIGHT
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols * unit_width)

    # Results
    tasks = df["task_name"].unique()
    for i, models in enumerate(LLM_FAMILIES.values()):
        n_models = len(models)
        total_units = ncols * unit_width
        row_units = n_models * unit_width
        offset = (total_units - row_units) // 2
        for j, model_name in enumerate(models):
            start = offset + j * unit_width
            end = start + unit_width
            ax = fig.add_subplot(gs[i, start:end])
            for module in ["ln2", "rc2"]:
                subset = df[(df["model_name"] == model_name) & (df["module"] == module)].copy()
                values = []
                depth = []
                for task_name in tasks:
                    df_task = subset[subset.task_name == task_name]
                    values.append(df_task["accuracy"])
                    df_task = df_task.sort_values("layer")
                    n_layers = df_task["layer"].max()
                    if len(depth) == 0:
                        depth = df_task["layer"] / n_layers
                values = np.asarray(values)
                acc = values.mean(axis=0)
                ax.plot(depth, acc, label=module)

                # Mark the highest fc1 performance
                if module == "ln2":
                    depth = np.asarray(depth)
                    index = np.argmax(acc)
                    ax.scatter(
                        depth[index],
                        acc[index],
                        s=MARKER_SIZE,
                        color="black",
                    )

                # Visualization
                ax.grid(alpha=ALPHA_GRID, lw=1.3)
                ax.spines["left"].set_linewidth(1)
                ax.spines["right"].set_linewidth(1)
                ax.spines["top"].set_linewidth(1)
                ax.spines["bottom"].set_linewidth(1)
                ax.tick_params(axis="both", direction="out", length=5, width=1, labelsize=FONTSIZE)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Normalized depth")
                ax.set_title(model_name)
    sns.despine(fig, fig.axes[0], trim=True, right=True, offset=10)

    # Common legend
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"probing_depth_language_all_{filename}"
        save_plot(figname=figname)


def plot_model_size_language(filename: str, save: bool = False) -> None:
    r"""Evolution of gain as a funnction of model size in billions of parameters."""
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Figure
    figsize = 2 * WIDTH, 2 * HEIGHT
    fig = plt.figure(figsize=figsize)

    # Results
    for family, size_map in LLM_SIZES.items():
        sub = df[df["model_name"].isin(size_map.keys())].copy()
        sub["size"] = sub["model_name"].map(size_map)
        sub = sub.sort_values("size")
        color = GAIN_PALETTE[family.lower()]
        plt.plot(
            sub["size"],
            sub["gain"],
            linestyle="--",
            color="black",
            linewidth=1.3,
            zorder=1,
        )
        plt.scatter(
            sub["size"],
            sub["gain"],
            s=220,
            color=color,
            edgecolor="black",
            linewidth=1.3,
            label=family,
            zorder=2,
        )
    ax = fig.axes[0]
    ax.set_xlabel("Model size (B parameters)", fontsize=FONTSIZE)
    ax.set_ylabel("Gain over standard probing", fontsize=FONTSIZE)
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1, labelsize=FONTSIZE)
    ax.legend(frameon=True, fontsize=FONTSIZE, loc="lower right")
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    plt.tight_layout()
    if save:
        figname = f"model_size_language_{filename}"
        save_plot(figname=figname)


# ------------------------------------------------------------------------------
# LLMs table functions
# ------------------------------------------------------------------------------
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

    # Heuristic depth
    heuristic_depth = 0.6

    # Results
    rows = []
    groups = "model_name"
    for model_name, sub in df.groupby(groups):
        last_layer = sub.layer.max()
        standard = sub[(sub.layer == last_layer) & (sub.module == "rc2")].accuracy.iloc[0]
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
                    "strategy": "Heuristic",
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
    table = table.loc[["Standard", "Heuristic", "Oracle"]]
    table["Average"] = table.mean(axis=1)
    ordered_models = [m for models in LLM_FAMILIES.values() for m in models]
    ordered_models = [m for m in ordered_models if m in table.columns]
    table = table[ordered_models + ["Average"]]
    print(table.to_latex(float_format="%.2f", bold_rows=True, escape=False))


def table_oracle_recovery_language(filename: str) -> None:
    r"""
    Recover the table of oracle gain recovery (computed as gain / oracle_gain) averaged over all models.

    Strategies:
        - Best layer on rc2
        - Heuristic (60% depth on ln2)
        - Best layer on ln2
    """
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()
    df = df.groupby(["model_name", "layer", "module"])["accuracy"].mean().reset_index()

    # Keep LLM models
    all_models = [m for models in LLM_FAMILIES.values() for m in models]
    df = df[df["model_name"].isin(all_models)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Heuristic depth
    heuristic_depth = 0.6

    # Results
    order = ["Best RC2", "Heuristic", "Best LN2"]
    groups = "model_name"
    rows = []
    for key, sub in df.groupby(groups):
        last_layer = sub["layer"].max()

        # Standard
        baseline = sub[(sub.layer == last_layer) & (sub.module == "rc2")].accuracy.iloc[0]

        # Best RC2, last layer act, best act
        rc2_best = sub[sub.module == "rc2"].accuracy.max()
        heuristic_layer = int(heuristic_depth * last_layer)
        heuristic = sub[(sub.layer == heuristic_layer) & (sub.module == "ln2")].accuracy.iloc[0]
        ln2_best = sub[sub.module == "ln2"].accuracy.max()
        strategies = {
            "Best RC2": rc2_best,
            "Heuristic": heuristic,
            "Best LN2": ln2_best,
        }

        # Recover oracle recovery
        oracle = sub.accuracy.max()
        oracle_gain = oracle - baseline

        # Ignore negligible oracle gains, i..e, less that 0.5 accuracy point
        if oracle_gain < 0.5:
            continue

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

    # Pooled mean recovery per strategy, single-row table
    pooled_recovery = summary.groupby("strategy")["recovery"].mean().reindex(order)
    table = pd.DataFrame({"Oracle recovery (%)": pooled_recovery}).T
    latex_str = table.round(1).to_latex(
        caption="Average gain and oracle gain recovery (\\%) per strategy, pooled over all models.",
        label=f"tab:oracle_recovery_{filename}",
        column_format="l" + "c" * len(order),
        escape=False,
    )
    print(latex_str)


def lomo_depth_heuristic(filename: str) -> None:
    r"""
    Leave-one-model-out estimate of a fixed relative-depth heuristic
    (module fixed at ln2). For each model, the heuristic depth is the
    median relative depth of the joint optimum across all other models;
    recovered gain is then evaluated on the held-out model at that
    depth, averaged over all MTEB tasks, avoiding data leakage.
    """
    # Recover raw per-task results, then average over tasks per (model, layer, module)
    path = RESULT_DIR
    full_df = pd.read_csv(path / f"{filename}.csv")
    full_df = full_df.copy()
    full_avg = full_df.groupby(["model_name", "layer", "module"])["accuracy"].mean().reset_index()

    # Recover aggregated per-model analysis table
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()
    df["relative_depth"] = df["best_layer"] / df["n_layers"]

    # Keep LLM models
    all_models = [m for models in LLM_FAMILIES.values() for m in models]
    df = df[df["model_name"].isin(all_models)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    results = []
    for held_out in df["model_name"]:
        other = df[df["model_name"] != held_out]
        heuristic_depth = 0.6  # other["relative_depth"].median()
        row = df[df["model_name"] == held_out].iloc[0]
        n_layers = row["n_layers"]
        heuristic_layer = round(heuristic_depth * (n_layers - 1))

        sub = full_avg[
            (full_avg["model_name"] == held_out)
            & (full_avg["module"] == "ln2")
            & (full_avg["layer"] == heuristic_layer)
        ]

        heuristic_accuracy = float(sub["accuracy"].iloc[0])
        heuristic_gain = heuristic_accuracy - row["baseline_accuracy"]

        results.append(
            {
                "model_name": held_out,
                "heuristic_depth": heuristic_depth,
                "heuristic_layer": heuristic_layer,
                "heuristic_gain": heuristic_gain,
                "oracle_gain": row["gain"],
                "recovery": heuristic_gain / row["gain"],
            }
        )
    results = pd.DataFrame(results)
    print(results)


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
    # plot_dataset_difficulty_vision(filename=filename, save=save)

    # # Cifar10-C
    # save = True
    # filename = "cifar10_c"
    # plot_cifar10c_severity(filename=filename, save=save)

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
    # plot_probing_depth_all_vision(filename=filename, save=save)

    # # Oracle recovery
    # save = True
    # filename = "cls"
    # plot_oracle_recovery_vision(filename=filename, save=save)

    r"""LLMs plots."""
    # # CDF
    # save = True
    # plot_cdf_language(filename="mteb_detailed", save=save)

    # # Model family
    # save = True
    # filename = "mteb"
    # plot_model_family_language(filename=filename, save=save)

    # Probing depth
    save = True
    filename = "mteb"
    plot_probing_depth_language(filename=filename, save=save)
    # plot_probing_depth_all_language(filename=filename, save=save)

    # # Model size
    # save = True
    # filename = "mteb"
    # plot_model_size_language(filename=filename, save=save)


def get_tables() -> None:
    r"""VFMs tables."""
    # # Model size
    # filename = "cls"
    # table_model_size_vision(filename=filename)

    # # Winning module frequency
    # filename = "cls"
    # table_winning_module_frequency_vision(filename=filename)

    # # Performance comparions
    # filename = "cls"
    # table_performance_comparison_vision(filename=filename)
    # table_performance_comparison_all_vision(filename=filename)

    r"""LLMs tables."""
    # Performance comparions
    filename = "mteb"
    table_performance_comparison_language(filename=filename)

    # Oracle recovery
    filename = "mteb"
    table_oracle_recovery_language(filename=filename)


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
