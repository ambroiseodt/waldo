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
VFMs = ["aim", "dino", "vit"]
VLMs = ["clip", "siglip"]
LLMs = ["gemma", "llama", "qwen"]
REPRESENTATIVE_MODELS = {"vit": "large", "dinov2": "large", "aimv2": "large", "dinov3": "vitl16"}
REPRESENTATIVE_DATASETS = ["aircraft", "domainnet-sketch", "food101", "flowers102"]
MODEL_SIZES = {
    "vit": ["base", "large", "huge"],
    "dinov2": ["small", "base", "large", "giant"],
    "aimv2": ["large", "huge", "1B", "3B"],
    "dinov3": ["vits16", "vitb16", "vitl16", "vit7b16"],
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
PALETTE = {
    "vit": "#4C72B0",
    "dinov2": "#DD8452",
    "aimv2": "#C44E52",
    "dinov3": "#55A868",
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
# Vision plotting functions
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
    print(f"Results are presented over a total of {len(df)} runs.")

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
    for model, size in REPRESENTATIVE_MODELS.items():
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
    print(f"Results are presented over a total of {len(df)} runs.")

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
    fig, ax = plt.subplots(figsize=figsize)

    # Results
    freq = df["best_module"].value_counts(normalize=True).mul(100).sort_values(ascending=False)
    sns.barplot(
        x=freq.index,
        y=freq.values,
        color=sns.color_palette()[0],
        edgecolor="black",
        linewidth=1,
        ax=ax,
        legend=False,
    )
    for i, v in enumerate(freq.values):
        ax.text(
            i,
            v + 1,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE,
        )

    # Visualization
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1, labelsize=FONTSIZE)
    ax.set_ylabel("Optimal module frequency (%)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        figname = f"winning_module_frequency_vision_{filename}"
        save_plot(figname=figname)


def plot_oracle_gain_vision(filename: str, save: bool = False) -> None:
    """Plot the gain over standard probing by optimizing both the probing module and layer."""

    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep only vision foundation models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Figure
    figsize = 2 * WIDTH, 2 * HEIGHT
    fig, ax = plt.subplots(figsize=figsize)

    # Results
    groups = ["dataset_name", "model_name", "model_size"]
    rows = []
    for key, sub in df.groupby(groups):
        last_layer = sub["layer"].max()
        baseline = sub[(sub.layer == last_layer) & (sub.module == "rc2")]["accuracy"].iloc[0]
        oracle = sub.sort_values("accuracy").iloc[-1]["accuracy"]
        rows.append({"family": key[1], "gain": oracle - baseline})
    gains = pd.DataFrame(rows)
    summary = gains.groupby("family", as_index=False).agg(gain=("gain", "mean")).sort_values("gain", ascending=False)
    sns.barplot(
        data=summary,
        x="family",
        y="gain",
        color="steelblue",
        edgecolor="black",
        linewidth=1,
        ax=ax,
    )

    # values above bars
    for i, v in enumerate(summary["gain"]):
        ax.text(
            i,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE - 2,
        )

    # Visualization
    ax.grid(axis="y", alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1, labelsize=FONTSIZE)
    ax.set_xlabel("")
    ax.set_ylabel("Oracle improvement (accuracy points)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        figname = f"oracle_gain_vision_{filename}"
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
    print(f"Results are presented over a total of {len(df)} runs.")

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_MODELS.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = df[mask]

    # Figure
    ncols = len(REPRESENTATIVE_MODELS)
    figsize = ncols * 2 * WIDTH, 2 * HEIGHT
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize)

    # Zoomed window (x and y axis are padded to avoid overlapping splines)
    ZOOM_XLIM = [(0.9, 1.0), (0.9, 1.0), (0.9, 1.0), (0.9, 1.0)]
    ZOOM_YLIM = [(44, 52), (72, 77), (77, 82), (75, 80)]
    positions = (0.55, 0.15)
    format = [(0.15, 0.25), (0.3, 0.2), (0.3, 0.2), (0.3, 0.2)]

    # Results
    dataset_name = "domainnet-sketch"
    for i, model_name in enumerate(REPRESENTATIVE_MODELS):
        ax = axes[i]
        axins = None
        if i:
            axins = ax.inset_axes((*positions, *format[i]))
        model_size = REPRESENTATIVE_MODELS[model_name]
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


def plot_probing_depth_vision_all(filename: str, save: bool = False) -> None:
    r"""Evolution of the act and rc2 probing accuracy along with the depth."""
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    print(f"Results are presented over a total of {len(df)} runs.")

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_MODELS.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = df[mask]

    # Figure
    datasets = df["dataset_name"].unique()
    nrows = len(datasets)  # len(REPRESENTATIVE_DATASETS)
    ncols = len(REPRESENTATIVE_MODELS)
    figsize = ncols * 2 * WIDTH, nrows * 2 * HEIGHT
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Results
    for i, dataset_name in enumerate(datasets):
        for j, model_name in enumerate(REPRESENTATIVE_MODELS):
            ax = axes[i, j]
            model_size = REPRESENTATIVE_MODELS[model_name]
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
        figname = f"probing_depth_vision_all_{filename}"
        save_plot(figname=figname)


def compare_heuristic_vs_standard(filename: str, save: bool = False) -> None:

    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    print(f"Results are presented over a total of {len(df)} runs.")

    group_cols = [
        "dataset_name",
        "model_name",
        "model_size",
        "patch_size",
        "resolution",
    ]

    rows = []

    for keys, sub in df.groupby(group_cols):
        dataset_name, model_name, model_size, patch_size, resolution = keys

        n_layers = sub["layer"].max() + 1

        # Standard probing: final RC2
        standard = sub[(sub["layer"] == sub["layer"].max()) & (sub["module"] == "rc2")]

        if standard.empty:
            continue

        standard_acc = standard["accuracy"].iloc[0]

        # Heuristic probing: ACT at family-specific normalized depth
        heuristic_depths = {
            "aimv2": 0.988,  # 0.8902173913043478,
            "dinov2": 0.988,  # 0.9213970811796898,
            "dinov3": 0.988,  # 0.8966580520928347,
            "vit": 0.938,  # 0.7475540821964385,
        }

        heuristic_layer = int(heuristic_depths[model_name] * (n_layers - 1))

        heuristic = sub[(sub["layer"] == heuristic_layer) & (sub["module"] == "act")]

        if heuristic.empty:
            continue

        heuristic_acc = heuristic["accuracy"].iloc[0]

        rows.append(
            {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "model_size": model_size,
                "standard_acc": standard_acc,
                "heuristic_acc": heuristic_acc,
                "gain": heuristic_acc - standard_acc,
            }
        )

    comparison = pd.DataFrame(rows)

    print(f"Number of points: {len(comparison)}")
    print(f"Mean gain: {comparison['gain'].mean():.3f}")
    print(f"Median gain: {comparison['gain'].median():.3f}")
    print(f"Improved cases: {(comparison['gain'] > 0).mean() * 100:.1f}%")

    # Scatter plot
    plt.figure(figsize=(6, 6))

    plt.scatter(comparison["standard_acc"], comparison["heuristic_acc"], alpha=0.7)

    lims = [
        min(comparison["standard_acc"].min(), comparison["heuristic_acc"].min()),
        max(comparison["standard_acc"].max(), comparison["heuristic_acc"].max()),
    ]

    plt.plot(lims, lims, "--")

    plt.xlabel("Standard probing (RC2)")
    plt.ylabel("Heuristic probing (ACT)")
    plt.title("Heuristic vs standard probing")

    plt.xlim(lims)
    plt.ylim(lims)

    # Common legend
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"heuristic_vs_standard_{filename}"
        save_plot(figname=figname)


def plot_act_gain_vision(
    filename: str,
    save: bool = False,
) -> None:
    """
    Plot the average gain obtained by the recommended probing strategy
    (ACT + best layer) over standard probing.
    """

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    STANDARD_MODULE = "rc2"
    BEST_MODULE = "act"

    groups = ["dataset_name", "model_name", "model_size"]

    rows = []

    for key, sub in df.groupby(groups):
        last_layer = sub["layer"].max()

        # Standard probing
        baseline = sub[(sub.layer == last_layer) & (sub.module == STANDARD_MODULE)]["accuracy"]

        if len(baseline) == 0:
            continue

        baseline = baseline.iloc[0]

        # Recommended strategy = ACT + best layer
        act_best = sub[sub.module == BEST_MODULE].sort_values("accuracy").iloc[-1]["accuracy"]

        rows.append(
            {
                "family": key[1],
                "gain": act_best - baseline,
            }
        )

    gains = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # Average over all configurations
    # ------------------------------------------------------------
    summary = (
        gains.groupby("family", as_index=False)
        .agg(
            gain=("gain", "mean"),
        )
        .sort_values("gain", ascending=False)
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    figsize = WIDTH, HEIGHT
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=summary,
        x="family",
        y="gain",
        color="steelblue",
        edgecolor="black",
        linewidth=1,
        ax=ax,
    )

    # values above bars
    for i, v in enumerate(summary["gain"]):
        ax.text(
            i,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE - 2,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Average location gain (%)", fontsize=FONTSIZE)

    ax.grid(axis="y", alpha=ALPHA_GRID)

    ax.tick_params(
        axis="both",
        direction="out",
        length=5,
        width=1,
    )

    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        figname = f"act_gain_vision_{filename}"
        save_plot(figname=figname)


def plot_layer_gain_vision(
    filename: str,
    save: bool = False,
) -> None:
    """
    Plot the average gain obtained by the recommended probing strategy
    (ACT + best layer) over standard probing.
    """

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    STANDARD_MODULE = "rc2"
    BEST_MODULE = "act"

    groups = ["dataset_name", "model_name", "model_size"]

    rows = []

    for key, sub in df.groupby(groups):
        last_layer = sub["layer"].max()

        # Standard probing
        baseline = sub[(sub.layer == last_layer) & (sub.module == STANDARD_MODULE)]["accuracy"]

        if len(baseline) == 0:
            continue

        baseline = baseline.iloc[0]

        # Recommended strategy = ACT + last layer
        act_best = sub[sub.module == "rc2"].sort_values("accuracy").iloc[-1]["accuracy"]

        rows.append(
            {
                "family": key[1],
                "gain": act_best - baseline,
            }
        )

    gains = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # Average over all configurations
    # ------------------------------------------------------------
    summary = (
        gains.groupby("family", as_index=False)
        .agg(
            gain=("gain", "mean"),
        )
        .sort_values("gain", ascending=False)
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    figsize = WIDTH, HEIGHT
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=summary,
        x="family",
        y="gain",
        color="steelblue",
        edgecolor="black",
        linewidth=1,
        ax=ax,
    )

    # values above bars
    for i, v in enumerate(summary["gain"]):
        ax.text(
            i,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE - 2,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Average location gain (%)", fontsize=FONTSIZE)

    ax.grid(axis="y", alpha=ALPHA_GRID)

    ax.tick_params(
        axis="both",
        direction="out",
        length=5,
        width=1,
    )

    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        figname = f"layer_gain_vision_{filename}"
        save_plot(figname=figname)


# ------------------------------------------------------------------------------
# Vision table functions
# ------------------------------------------------------------------------------
def table_model_size(filename: str) -> None:
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
    for family, sizes in MODEL_SIZES.items():
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


def table_oracle_recovery(filename: str) -> None:
    r"""
    Analyze the oracle performance recovery of different probing strategies.

    The different probing strategies are:
        - standard: probing of rc2 module at the last layer
        - best rc2: best probing of rc2 over the depth
        - best act: best probing of act over the depth
    """
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Results
    groups = ["dataset_name", "model_name", "model_size"]
    rows = []
    for key, sub in df.groupby(groups):
        # Standard
        last_layer = sub["layer"].max()
        baseline = sub[(sub["layer"] == last_layer) & (sub["module"] == "rc2")]["accuracy"]
        baseline = baseline.iloc[0]

        # Best rc2
        rc2_best = sub[sub.module == "rc2"].sort_values("accuracy").accuracy.iloc[-1]

        # Last layer act
        last_layer = sub["layer"].max()
        act_last = sub[(sub["layer"] == last_layer) & (sub["module"] == "act")]["accuracy"]
        act_last = act_last.iloc[0]

        # Best act
        act_best = sub[sub.module == "act"].sort_values("accuracy").accuracy.iloc[-1]

        # Oracle
        oracle = sub["accuracy"].max()
        oracle_gain = oracle - baseline
        if oracle_gain < 1e-8:
            continue

        # Table
        strategies = {
            "Standard": baseline,
            "Best layer + rc2": rc2_best,
            "Last layer + act": act_last,
            "Best layer + act": act_best,
        }

        for strategy, acc in strategies.items():
            gain = acc - baseline
            recovered = np.clip(gain / oracle_gain, 0, 1)
            exact = np.isclose(gain, oracle_gain)
            rows.append(
                {
                    "dataset": key[0],
                    "family": key[1],
                    "size": key[2],
                    "strategy": strategy,
                    "gain": gain,
                    "recovered": recovered * 100,
                    "exact": exact,
                }
            )

    summary = pd.DataFrame(rows)
    order = [
        "Standard",
        "Best layer + rc2",
        "Last layer + act",
        "Best layer + act",
    ]
    family = (
        summary.groupby(["family", "strategy"])
        .agg(recovery=("recovered", "mean"), optimum=("exact", "mean"))
        .reset_index()
    )
    family["optimum"] *= 100
    recovery_family = family.pivot(index="family", columns="strategy", values="recovery").reindex(columns=order)
    optimum_family = family.pivot(index="family", columns="strategy", values="optimum").reindex(columns=order)
    overall = summary.groupby("strategy").agg(
        recovery=("recovered", "mean"),
        optimum=("exact", "mean"),
    )
    overall["optimum"] *= 100
    overall_recovery = overall["recovery"].reindex(order).to_frame().T
    overall_recovery.index = ["Overall"]
    overall_optimum = overall["optimum"].reindex(order).to_frame().T
    overall_optimum.index = ["Overall"]
    recovery = pd.concat([recovery_family, overall_recovery])
    optimum = pd.concat([optimum_family, overall_optimum])
    table = pd.concat({"Recovery (%)": recovery, "Optimum (%)": optimum}, axis=1)
    print(table)


def compute_depth_region_frequency(filename: str) -> None:
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    tolerance = 0.01
    n_bins = 20

    df = df[df["module"] == "act"].copy()

    group_cols = [
        "dataset_name",
        "model_name",
        "model_size",
        "patch_size",
        "resolution",
    ]

    results = []

    for keys, sub in df.groupby(group_cols):
        # normalize depth
        max_layer = sub["layer"].max()
        sub = sub.copy()
        sub["depth"] = sub["layer"] / max_layer

        # oracle activation accuracy for this model/dataset
        best_acc = sub["accuracy"].max()

        # whether each layer is within tolerance of oracle
        sub["near_optimal"] = sub["accuracy"] >= best_acc * (1 - tolerance)

        # assign depth bins
        gap = 50 // n_bins
        sub["depth_bin"] = pd.cut(
            sub["depth"],
            bins=np.linspace(0.5, 1, n_bins + 1),
            # labels=[f"{int(i * gap)}-{int((i + 1) * gap)}%" for i in range(n_bins)],
            include_lowest=True,
        )

        # one value per depth bin: whether this region contains
        # a near-optimal layer
        bin_values = sub.groupby("depth_bin", observed=False)["near_optimal"].any()

        results.append({"model_name": keys[1], **bin_values.to_dict()})

    result = pd.DataFrame(results)

    # convert model_name to model family
    result = result.groupby("model_name").mean(numeric_only=True) * 100
    print(result)


def table_depth(filename: str, verbose: bool = False) -> dict:
    r"""Compute the median optimal depth to choose for the act module."""
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep only vision foundation models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
    df = df.copy()

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_MODELS.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = df[mask]

    # Results
    df["n_layers"] = df.groupby(["model_name", "model_size"])["layer"].transform("max")
    df["depth"] = df["layer"] / df["n_layers"]
    df = df[df["module"] == "act"]
    results = []
    for (model_name, model_size, dataset_name), g in df.groupby(["model_name", "model_size", "dataset_name"]):
        best = g.loc[g["accuracy"].idxmax()]
        results.append(
            {
                "model_name": model_name,
                "model_size": model_size,
                "dataset_name": dataset_name,
                "best_layer": best["layer"],
                "n_layers": best["n_layers"],
                "best_depth": best["depth"],
                "best_accuracy": best["accuracy"],
            }
        )
    df = pd.DataFrame(results)
    depth = (
        df.groupby(["model_name", "model_size", "dataset_name"])
        .agg(mean_best_depth=("best_depth", "mean"))
        .reset_index()
    )
    if verbose:
        depth["mean_best_depth"] = 100 * depth["mean_best_depth"]
        print(depth)
    depths = {row["model_name"]: row["mean_best_depth"] for idx, row in depth.iterrows()}
    return depths


def table_gains(filename: str) -> None:
    r"""
    Analyze the oracle performance recovery and frequency of optimality of different probing strategies.

    The different probing strategies are:
        - standard: probing of rc2 module at the last layer
        - best rc2: best probing of rc2 over the depth
        - fixed act: probing of act module at the layer predicted by the depth analysis
        - best act: best probing of act over the depth
    """
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # Recover the heuristic depth for each model family
    depths = {
        "aimv2": 1,  # 0.8902173913043478,
        "dinov2": 1,  # 0.9213970811796898,
        "dinov3": 0.97,  # 0.96,  # 0.8966580520928347,
        "vit": 0.72,  # 0.7475540821964385,
    }  # analyze_depth(filename=filename, verbose=False)
    print(depths)
    df["heuristic_depth"] = df["model_name"].map(depths)

    # Results
    groups = ["dataset_name", "model_name", "model_size"]
    rows = []
    for key, sub in df.groupby(groups):
        # Standard
        last_layer = sub["layer"].max()
        rc2_std = sub[(sub["layer"] == last_layer) & (sub["module"] == "rc2")]["accuracy"]
        rc2_std = rc2_std.iloc[0]

        # Best rc2
        rc2_best = sub[sub.module == "rc2"].sort_values("accuracy").accuracy.iloc[-1]

        # Fixed act
        heuristic_depth = sub["heuristic_depth"].iloc[0]
        heuristic_layer = int(heuristic_depth * last_layer)
        act_fixed = sub[(sub["layer"] == heuristic_layer) & (sub["module"] == "act")]["accuracy"]
        act_fixed = act_fixed.iloc[0] if len(act_fixed) else np.nan

        # Best act
        act_best = sub[sub.module == "act"].sort_values("accuracy").accuracy.iloc[-1]

        # Oracle
        oracle_acc = sub["accuracy"].max()
        oracle_gain = oracle_acc - rc2_std
        if oracle_gain < 1e-8:
            continue

        # Table
        strategies = {
            "Standard": rc2_std,
            "Best layer + rc2": rc2_best,
            "Fixed layer + act": act_fixed,
            "Best layer + act": act_best,
        }

        for strategy, acc in strategies.items():
            gain = acc - rc2_std
            recovered = np.clip(gain / oracle_gain, 0, 1)
            exact = np.isclose(gain, oracle_gain)
            rows.append(
                {
                    "dataset": key[0],
                    "family": key[1],
                    "size": key[2],
                    "strategy": strategy,
                    "gain": gain,
                    "recovered": recovered * 100,
                    "exact": exact,
                }
            )

    summary = pd.DataFrame(rows)
    print(summary)
    # ============================================================
    # Final table
    # ============================================================

    order = [
        "Standard",
        "Best layer + rc2",
        "Fixed layer + act",
        "Best layer + act",
    ]

    # ------------------------------------------------------------
    # Family table
    # ------------------------------------------------------------
    family = (
        summary.groupby(["family", "strategy"])
        .agg(
            gain=("gain", "mean"),
            recovery=("recovered", "mean"),
            optimum=("exact", "mean"),
        )
        .reset_index()
    )

    family["optimum"] *= 100

    gain_family = family.pivot(index="family", columns="strategy", values="gain").reindex(columns=order)
    recovery_family = family.pivot(index="family", columns="strategy", values="recovery").reindex(columns=order)
    optimum_family = family.pivot(index="family", columns="strategy", values="optimum").reindex(columns=order)

    # ------------------------------------------------------------
    # Overall row
    # ------------------------------------------------------------
    overall = summary.groupby("strategy").agg(
        gain=("gain", "mean"),
        recovery=("recovered", "mean"),
        optimum=("exact", "mean"),
    )
    overall["optimum"] *= 100
    overall_gain = overall["gain"].reindex(order).to_frame().T
    overall_gain.index = ["Overall"]
    overall_recovery = overall["recovery"].reindex(order).to_frame().T
    overall_recovery.index = ["Overall"]
    overall_optimum = overall["optimum"].reindex(order).to_frame().T
    overall_optimum.index = ["Overall"]

    # ------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------
    gain = pd.concat([gain_family, overall_gain])
    recovery = pd.concat([recovery_family, overall_recovery])
    optimum = pd.concat([optimum_family, overall_optimum])

    table = pd.concat(
        {
            "Gain": gain,
            "Recovery (%)": recovery,
            "Optimum (%)": optimum,
        },
        axis=1,
    )

    table = table.swaplevel(0, 1, axis=1)
    table = table.sort_index(axis=1, level=0)

    table = table.round(1)

    print(table)

    # Latex
    print(
        table.to_latex(
            multicolumn=True,
            multirow=True,
            float_format="%.1f",
        )
    )


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

    # Keep VLMs rows from dataframe
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


def plot_layer_winning_frequency_vision(
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

    # Figure
    figsize = 2 * WIDTH, 2 * HEIGHT
    fig, ax = plt.subplots(figsize=figsize)

    # Results
    freq = df["best_layer"].value_counts(normalize=True).mul(100).sort_values(ascending=False)
    sns.barplot(
        x=freq.index,
        y=freq.values,
        color=sns.color_palette()[0],
        edgecolor="black",
        linewidth=1,
        ax=ax,
        legend=False,
    )
    for i, v in enumerate(freq.values):
        ax.text(
            i,
            v + 1,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE,
        )

    # Visualization
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1, labelsize=FONTSIZE)
    ax.set_ylabel("Optimal module frequency (%)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        figname = f"winning_frequency_language_{filename}"
        save_plot(figname=figname)


# ------------------------------------------------------------------------------
# Results functions
# ------------------------------------------------------------------------------
def plot_figures() -> None:
    # # CDF
    # save = True
    # plot_cdf_vision(filename="cls", save=save)
    # plot_cdf_multimodal(filename="cls", save=save)
    # plot_cdf_language(filename="mteb", save=save)

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

    # # Oracle gain
    # save = True
    # filename = "cls"
    # plot_oracle_gain_vision(filename=filename, save=save)

    # # Probing depth
    # save = True
    # filename = "cls"
    # plot_probing_depth_vision(filename=filename, save=save)
    # plot_probing_depth_vision_all(filename=filename, save=save)

    # Heuristic
    save = True
    filename = "cls"
    compare_heuristic_vs_standard(filename=filename, save=save)

    # # Location gain
    # save = True
    # filename = "cls"
    # plot_act_gain_vision(filename=filename, save=save)

    # # Module gain
    # save = True
    # filename = "cls"
    # plot_layer_gain_vision(filename=filename, save=save)

    # # Layer winning frequency
    # save = True
    # filename = "mteb"
    # plot_layer_winning_frequency_vision(filename=filename, save=save)


def get_tables() -> None:
    # # Model size
    # filename = "cls"
    # table_model_size(filename=filename)

    # # Oracle recovery
    # filename = "cls"
    # table_oracle_recovery(filename=filename)

    # Heuristic depth
    filename = "cls"
    compute_depth_region_frequency(filename=filename)

    # Module depth
    filename = "cls"
    table_depth(filename=filename, verbose=True)

    # # Gains
    # filename = "cls"
    # table_gains(filename=filename)


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
