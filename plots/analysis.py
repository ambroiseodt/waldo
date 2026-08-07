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
from scipy.stats import friedmanchisquare, spearmanr

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
MODULE_ORDER = ["ln1", "mha", "rc1", "ln2", "fc1", "act", "fc2", "rc2"]
VFMs = ["aim", "dino", "vit"]
REPRESENTATIVE_VFMs = {"vit": "large", "dinov2": "large", "aimv2": "large", "dinov3": "vitl16"}
VFM_SIZES = {
    "vit": ["base", "large", "huge"],
    "dinov2": ["small", "base", "large", "giant"],
    "aimv2": ["large", "huge", "1B", "3B"],
    "dinov3": ["vits16", "vitb16", "vitl16", "vit7b16"],
}
REPRESENTATIVE_DATASETS = ["aircraft", "domainnet-sketch", "food101", "flowers102"]
LLMs = ["qwen3", "llama3", "gemma3"]
REPRESENTATIVE_LLMs = {"qwen3": "Qwen3-1.7B", "llama3": "Llama-3.2-1B", "gemma3": "gemma-3-1b-pt"}
LLM_FAMILIES = {
    "qwen3": ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"],
    "llama3": ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B"],
    "gemma3": ["gemma-3-1b-pt", "gemma-3-4b-pt"],
}
LLM_SIZES = {
    "Qwen3": {"Qwen3-1.7B": 1.7, "Qwen3-4B": 4, "Qwen3-8B": 8},
    "Llama3": {"Llama-3.2-1B": 1, "Llama-3.2-3B": 3, "Llama-3.1-8B": 8},
    "Gemma3": {"gemma-3-1b-pt": 1, "gemma-3-4b-pt": 4},
}
VLMs = ["clip", "siglip"]
REPRESENTATIVE_VLMS = {"clip": "large", "siglip2": "large"}

# Plot aesthetic
WIDTH = 4
HEIGHT = WIDTH
FONTSIZE = 19
LEGFONTSIZE = 10
LINEWIDTH = 3
BARWIDTH = 0.25
MARKER_SIZE = 150
ALPHA_GRID = 0.8
PALETTE = {
    "vit": "#4C72B0",
    "dinov2": "#DD8452",
    "aimv2": "#C44E52",
    "dinov3": "#55A868",
    "qwen3": "#7B5EA7",
    "llama3": "#F0A860",
    "gemma3": "#4C72B0",
}


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
def plot_cdf_vision(filename: str, save: bool = False) -> None:
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
    save: bool, default=False
        Whether to save the plot.
    """
    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Figure
    figsize = 2 * WIDTH, 2 * HEIGHT
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    # Zoomed window (x and y axis are padded to avoid overlapping splines)
    ZOOM_XLIM = (-0.05, 0.2)
    ZOOM_YLIM = (80, 100)
    axins = ax.inset_axes((0.25, 0.48, 0.24, 0.44))

    # Results
    for col, label in [("gain_layer", "Layer"), ("gain_module", "Module"), ("gain", "Location")]:
        gains = np.sort(df[col].to_numpy())
        percentage = 100 * (1 - np.arange(1, len(gains) + 1) / len(gains))
        ax.plot(gains, percentage, linewidth=LINEWIDTH, label=label)
        axins.plot(gains, percentage, linewidth=LINEWIDTH)

    # Subplot limits
    axins.set_xlim(*ZOOM_XLIM)
    axins.set_ylim(*ZOOM_YLIM)
    axins.grid(alpha=ALPHA_GRID, lw=1.0)
    axins.tick_params(axis="both", direction="out", length=3, width=1, labelsize=FONTSIZE * 0.6)
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
    ax.set_xlabel("Gain over standard probing", fontsize=FONTSIZE)
    ax.set_ylabel(r"Percentage of runs with gain $> x$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    ax.legend()
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

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

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
    ax.set_ylabel("Mean probing gain", fontsize=FONTSIZE)
    ax.set_title(f"Dataset difficulty vs gain\nSpearman ρ = {rho:.2f}, p = {pval:.2e}", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"dataset_difficulty_vision_{filename}"
        save_plot(figname=figname)


def plot_cifar10c_severity(filename: str, save: bool = False) -> None:
    r"""
    Plot probing gain as a function of corruption severity on CIFAR10-C.

    Each curve corresponds to one representative vision model.
    Results are averaged over corruption types.
    """

    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Figure
    figsize = 2 * WIDTH, 2 * HEIGHT
    fig = plt.figure(figsize=figsize)

    # Results
    severity_order = [1, 3, 5]
    sns.barplot(
        data=df,
        x="severity",
        y="gain",
        order=severity_order,
        estimator=np.mean,
        errorbar=("ci", 95),
        capsize=0.15,
        color="steelblue",
    )

    # Visualization
    ax = fig.axes[0]
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xlabel("Corruption severity", fontsize=FONTSIZE)
    ax.set_ylabel("Mean probing gain", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        save_plot(figname=f"cifar10c_severity_{filename}")


def plot_cifar10c_severity_all(
    filename: str,
    save: bool = False,
) -> None:
    """
    Plot probing gain as a function of corruption severity
    for each CIFAR10-C corruption separately.

    Each curve is averaged over
        - model families
        - model sizes
    """

    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv").copy()

    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    severity_order = [1, 3, 5]
    corruptions = sorted(df["corruption"].unique())

    # Figure
    ncols = 5
    nrows = int(np.ceil(len(corruptions) / ncols))
    figsize = (ncols * WIDTH, nrows * HEIGHT)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    # Results
    axes = np.asarray(axes).flatten()
    color = sns.color_palette()[0]
    for i, (ax, corruption) in enumerate(zip(axes, corruptions, strict=True)):
        sub = df[df["corruption"] == corruption]
        stats = sub.groupby("severity")["gain"].mean().reindex(severity_order)
        ax.plot(
            severity_order,
            stats.values,
            color=color,
            marker="o",
            lw=2,
            ms=5,
        )

        # Visualization
        ax.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1)
        ax.set_xlabel("Corruption severity", fontsize=FONTSIZE)
        if i == 0:
            ax.set_ylabel("Mean probing gain", fontsize=FONTSIZE)
        ax.set_title(corruption, fontsize=FONTSIZE - 2)
    sns.despine(fig, axes[0], trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        save_plot(f"cifar10c_severity_all_{filename}")


def plot_pretraining_scale_vision(filename: str, save: bool = False) -> None:
    r"""
    Plot the Mean probing gain for different pretraining scales.

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

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_VFMs.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = df[mask]

    # Pretraining scale
    pretraining_tokens = {
        "vit": 2.5e11,
        "dinov2": 1.80e12,
        "aimv2": 3.15e12,
        "dinov3": 3.91e12,
    }
    token_labels = {"vit": "250B", "dinov2": "1.8T", "aimv2": "3.2T", "dinov3": "3.9T"}
    df["pretraining_tokens"] = df["model_name"].map(pretraining_tokens)

    # Figure
    figsize = 2 * WIDTH, 2 * HEIGHT
    fig = plt.figure(figsize=figsize)

    # Results
    stats = {}
    models = ["vit", "dinov2", "aimv2", "dinov3"]
    for model in models:
        vals = df[df["model_name"] == model]["gain"].to_numpy()
        stats[model] = {
            "median": np.median(vals),
            "q25": np.percentile(vals, 25),
            "q75": np.percentile(vals, 75),
        }

    scales = np.arange(len(models))
    medians = [stats[model]["median"] for model in models]
    q25s = [stats[model]["q25"] for model in models]
    q75s = [stats[model]["q75"] for model in models]
    plt.plot(
        scales,
        medians,
        color="#2D3A8C",
        lw=1.8,
        zorder=3,
        linestyle="--",
        alpha=0.5,
    )
    plt.fill_between(
        scales,
        q25s,
        q75s,
        color="#2D3A8C",
        alpha=0.15,
        zorder=2,
        label="IQR across datasets (Q1–Q3)",
    )

    # One coloured marker per model family
    models = ["vit", "dinov2", "aimv2", "dinov3"]
    for scale, model in zip(scales, models, strict=True):
        plt.scatter(
            scale,
            stats[model]["median"],
            color=PALETTE[model],
            s=100,
            zorder=4,
            edgecolors="white",
            linewidths=1.2,
            label=model,
        )

    # Visualization
    ax = fig.axes[0]
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    ax.set_xticks(scales)
    ax.set_xticklabels([f"{model}\n{token_labels[model]}" for model in models], fontsize=FONTSIZE)
    ax.set_ylabel("Probing gain", fontsize=FONTSIZE)
    ax.set_xlabel("Model family  /  Pretraining scale", fontsize=FONTSIZE)
    ax.set_xlim(-0.4, len(models) - 0.6)
    ax.set_ylim(bottom=0)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.legend()
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

    # Figure
    N = len(REPRESENTATIVE_DATASETS)
    figsize = 2 * N * WIDTH, 2 * HEIGHT
    fig, axes = plt.subplots(1, N, figsize=figsize, sharey=True)
    model_order = list(REPRESENTATIVE_VFMs.keys())
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


def plot_winning_module_frequency_vision(
    filename: str,
    save: bool = False,
) -> None:
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

    # Figure
    figsize = 2 * WIDTH, 2 * HEIGHT
    fig = plt.figure(figsize=figsize)

    # Results
    freq = df["best_module"].value_counts(normalize=True).mul(100)
    freq = freq.reindex(MODULE_ORDER, fill_value=0.0).sort_values(ascending=False)
    ranks = np.arange(1, len(freq) + 1)
    sns.barplot(
        x=ranks,
        y=freq.values,
        color=sns.color_palette()[0],
        edgecolor="black",
        linewidth=1,
        legend=False,
    )

    # Annotate module name and value above each bar
    for i, (module, v) in enumerate(freq.items()):
        plt.text(
            i,
            v + 1,
            f"{module}",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE - 1,
            rotation=0,
        )

    # Visualization
    ax = fig.axes[0]
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1, labelsize=FONTSIZE)
    ax.set_xlabel("Rank", fontsize=FONTSIZE)
    ax.set_xticks(np.arange(len(freq)))
    ax.set_xticklabels(ranks)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        figname = f"winning_module_frequency_vision_{filename}"
        save_plot(figname=figname)


def plot_probing_depth_vision(filename: str, save: bool = False) -> None:
    r"""Evolution of the act and rc2 probing accuracy along with the depth."""
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

    # Figure
    ncols = len(REPRESENTATIVE_VFMs)
    figsize = ncols * 2 * WIDTH, 2 * HEIGHT
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize)

    # Zoomed window (x and y axis are padded to avoid overlapping splines)
    ZOOM_XLIM = [(0.9, 1.0), (0.9, 1.0), (0.9, 1.0), (0.9, 1.0)]
    ZOOM_YLIM = [(44, 52), (72, 77), (77, 82), (75, 80)]
    positions = (0.55, 0.15)
    format = [(0.15, 0.25), (0.3, 0.2), (0.3, 0.2), (0.3, 0.2)]

    # Results
    dataset_name = "domainnet-sketch"
    for i, model_name in enumerate(REPRESENTATIVE_VFMs):
        ax = axes[i]
        axins = None
        if i:
            axins = ax.inset_axes((*positions, *format[i]))
        model_size = REPRESENTATIVE_VFMs[model_name]
        acc_level = 0
        for module in ["act", "rc2"]:
            subset = df[
                (df["model_name"] == model_name)
                & (df["model_size"] == model_size)
                & (df["dataset_name"] == dataset_name)
                & (df["module"] == module)
            ].copy()
            subset = subset.sort_values("layer")
            n_layers = subset["layer"].max()
            subset["depth"] = subset["layer"] / n_layers
            ax.plot(subset["depth"], subset["accuracy"], linewidth=LINEWIDTH, label=module)

            # Zoomed-in regions
            if axins is not None:
                axins.plot(subset["depth"], subset["accuracy"], linewidth=LINEWIDTH)
                if acc_level < subset["accuracy"].values[-1]:
                    acc_level = subset["accuracy"].values[-1]

            # Mark the highest act performance
            if module == "act":
                index = np.argmax(subset["accuracy"])
                ax.scatter(
                    subset["depth"].values[index],
                    subset["accuracy"].values[index],
                    s=MARKER_SIZE,
                    color="black",
                )

        # Subplot limits
        if axins is not None:
            axins.set_xlim(*ZOOM_XLIM[i])
            axins.set_ylim(*ZOOM_YLIM[i])
            axins.grid(alpha=ALPHA_GRID, lw=1.0)
            axins.tick_params(axis="both", direction="out", length=3, width=1, labelsize=FONTSIZE * 0.6)
            for spine in axins.spines.values():
                spine.set_linewidth(0.8)
                spine.set_edgecolor("0.6")

            # Align zoomed window and subplot
            ax.indicate_inset_zoom(axins, edgecolor="0.15", linewidth=1.0, linestyle="--")

        # Visualization
        ax.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=5, width=1, labelsize=FONTSIZE)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Normalized depth")
        if i == 0:
            ax.set_ylabel("Accuracy")
        ax.set_title(model_name)
    sns.despine(fig, axes[0], trim=True, right=True, offset=10)

    # Common legend
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"probing_depth_vision_{filename}"
        save_plot(figname=figname)


def plot_probing_depth_all_vision(filename: str, save: bool = False) -> None:
    r"""Evolution of the act and rc2 probing accuracy along with the depth."""
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

    # Figure
    nrows = len(REPRESENTATIVE_DATASETS)
    ncols = len(REPRESENTATIVE_VFMs)
    figsize = ncols * 2 * WIDTH, nrows * 2 * HEIGHT
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

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
                subset["depth"] = subset["layer"] / n_layers
                ax.plot(subset["depth"], subset["accuracy"], linewidth=LINEWIDTH, label=module)

                # Mark the highest act performance
                if module == "act":
                    index = np.argmax(subset["accuracy"])
                    ax.scatter(
                        subset["depth"].values[index],
                        subset["accuracy"].values[index],
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
            if j == 0:
                ax.set_ylabel(f"{dataset_name} \n Accuracy")
            ax.set_title(model_name)
    sns.despine(fig, axes[0, 0], trim=True, right=True, offset=10)

    # Common legend
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"probing_depth_all_vision_{filename}"
        save_plot(figname=figname)


def plot_probing_depth_family_vision(filename: str, model_name: str, save: bool = False) -> None:
    r"""Evolution of the act and rc2 probing accuracy along with the depth."""
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Recover model family
    df = df[df["model_name"] == model_name]
    model_sizes = VFM_SIZES[model_name]

    # Figure
    nrows = len(model_sizes)
    ncols = len(REPRESENTATIVE_DATASETS)
    figsize = ncols * 2 * WIDTH, nrows * 2 * HEIGHT
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Results
    for i, model_size in enumerate(model_sizes):
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
                subset["depth"] = subset["layer"] / n_layers
                ax.plot(subset["depth"], subset["accuracy"], linewidth=LINEWIDTH, label=module)

                # Mark the highest act performance
                if module == "act":
                    index = np.argmax(subset["accuracy"])
                    ax.scatter(
                        subset["depth"].values[index],
                        subset["accuracy"].values[index],
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
            if i == 0:
                ax.set_title(f"{dataset_name} \n Accuracy")
            if j == 0:
                ax.set_ylabel(f"{model_name} - {model_size}")
    sns.despine(fig, axes[0, 0], trim=True, right=True, offset=10)

    # Common legend
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"probing_depth_family_{model_name}_vision_{filename}"
        save_plot(figname=figname)


# ------------------------------------------------------------------------------
# VFMs table functions
# ------------------------------------------------------------------------------
def table_model_size_vision(filename: str) -> None:
    r"""
    Friedman test to determine whether the model size has a significant impact on the gain
    per model family. The null hypothesis is that it influences the gain.
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
                "family": family,
                "n_sizes": k,
                "n_datasets": n,
                "chi2": stat,
                "p": p,
                "W": W,
            }
        )

    friedman_table = pd.DataFrame(results)
    print(friedman_table.to_latex(index=False, float_format="%.3f"))


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
                    "strategy": "Standard",
                    "accuracy": standard,
                },
                {
                    "dataset": dataset,
                    "family": family,
                    "strategy": "Heuristic",
                    "accuracy": heuristic,
                },
                {
                    "dataset": dataset,
                    "family": family,
                    "strategy": "Oracle",
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
    print(table.to_latex(float_format="%.2f", multirow=True, bold_rows=True))


def table_oracle_recovery_vision(filename: str) -> None:
    r"""
    Recover the table of oracle gain recovery (computed as gain / oracle_gain) averaged over all datasets and models.

    Strategies:
        - Best layer on rc2
        - Heuristic (last layer on act)
        - Best layer on act
    """

    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

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

    # Pooled mean recovery per strategy, single-row table
    pooled_recovery = summary.groupby("strategy")["recovery"].mean().reindex(order)
    table = pd.DataFrame({"Oracle recovery (%)": pooled_recovery}).T
    latex_str = table.round(1).to_latex(
        caption="Average gain and oracle gain recovery (\\%) per strategy, pooled over all datasets and models.",
        label=f"tab:oracle_recovery_{filename}",
        column_format="l" + "c" * len(order),
        escape=False,
    )
    print(latex_str)


# ------------------------------------------------------------------------------
# LLMs plotting functions
# ------------------------------------------------------------------------------
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
    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()
    logger.info(f"Results are presented over a total of {len(df)} models.")

    # Figure
    ncols = len(LLM_FAMILIES)
    figsize = ncols * 2 * WIDTH, 2 * HEIGHT
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize)
    gain_types = ["gain_layer", "gain_module", "gain"]
    gain_labels = ["Layer", "Module", "Both"]
    colors = sns.color_palette("Set2", 3)

    # Results
    for ax, (family_name, model_order) in zip(axes, LLM_FAMILIES.items(), strict=True):
        agg = df[df["model_name"].isin(model_order)].set_index("model_name").reindex(model_order)
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
        ax.set_title(family_name, fontsize=FONTSIZE)
        ax.set_xticks(x + BARWIDTH)
        ax.set_xticklabels(model_order, rotation=20)
        if ax == axes[0]:
            ax.set_ylabel("Gain over standard probing (MTEB avg.)", fontsize=FONTSIZE)
    sns.despine(fig, axes[0], trim=True, right=True, offset=10)

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
        figname = f"model_family_language_{filename}"
        save_plot(figname=figname)


def plot_probing_depth_language(filename: str, save: bool = False) -> None:
    r"""Evolution of the act and rc2 probing accuracy along with the depth."""
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Figure
    ncols = len(REPRESENTATIVE_LLMs)
    figsize = ncols * 2 * WIDTH, 2 * HEIGHT
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize)

    # Results
    tasks = df["task_name"].unique()
    for i, model_name in enumerate(REPRESENTATIVE_LLMs.values()):
        ax = axes[i]
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
    sns.despine(fig, axes[0], trim=True, right=True, offset=10)

    # Common legend
    plt.legend()
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
        color = PALETTE[family.lower()]
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
    # plot_cifar10c_severity_all(filename=filename, save=save)

    # # Pretraining scale
    # save = True
    # filename = "cls"
    # plot_pretraining_scale_vision(filename=filename, save=save)

    # # Model family
    # save = True
    # filename = "cls"
    # plot_model_family_vision(filename=filename, save=save)

    # # Module winning frequency
    # save = True
    # filename = "cls"
    # plot_winning_module_frequency_vision(filename=filename, save=save)

    # # Probing depth
    # save = True
    # filename = "cls"
    # plot_probing_depth_vision(filename=filename, save=save)
    # plot_probing_depth_all_vision(filename=filename, save=save)
    # for model_name in ["vit", "dinov2", "aimv2", "dinov3"]:
    #     plot_probing_depth_family_vision(filename=filename, model_name=model_name, save=save)

    r"""LLMs plots."""
    # Model family
    save = True
    filename = "mteb"
    plot_model_family_language(filename=filename, save=save)

    # Probing depth
    save = True
    filename = "mteb"
    plot_probing_depth_language(filename=filename, save=save)
    plot_probing_depth_all_language(filename=filename, save=save)

    # Model size
    save = True
    filename = "mteb"
    plot_model_size_language(filename=filename, save=save)


def get_tables() -> None:
    r"""VFMs tables."""
    # # Model size
    # filename = "cls"
    # table_model_size_vision(filename=filename)

    # # Performance comparions
    # filename = "cls"
    # table_performance_comparison_vision(filename=filename)

    # # Oracle recovery
    # filename = "cls"
    # table_oracle_recovery_vision(filename=filename)

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
