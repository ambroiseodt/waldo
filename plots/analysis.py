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
    ZOOM_XLIM = (-0.05, 0.35)
    ZOOM_YLIM = (75, 100)
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
    ax.set_ylabel("Median probing gain", fontsize=FONTSIZE)
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


def plot_module_depth_vision(filename: str, save: bool = False) -> None:
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
    REPRESENTATIVE_DATASETS = df["dataset_name"].unique()
    nrows = len(REPRESENTATIVE_DATASETS)
    ncols = len(REPRESENTATIVE_MODELS)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 3 * nrows),
        # sharex=True,
        # sharey="row",
    )
    for i, dataset in enumerate(REPRESENTATIVE_DATASETS):
        print(dataset)
        for j, model_name in enumerate(REPRESENTATIVE_MODELS):
            print(model_name)
            model_size = REPRESENTATIVE_MODELS[model_name]
            ax = axes[i, j]
            for module in ["act", "rc2"]:
                subset = df[
                    (df["model_name"] == model_name)
                    & (df["model_size"] == model_size)
                    & (df["dataset_name"] == dataset)
                    & (df["module"] == module)
                ].copy()

                if len(subset) == 0:
                    ax.set_axis_off()
                    continue

                subset = subset.sort_values("layer")

                n_layers = subset["layer"].max()

                subset["depth"] = subset["layer"] / n_layers

                ax.plot(subset["depth"], subset["accuracy"], linewidth=2, label=module)
                print(module)
                print(subset["accuracy"].values[-1])
            ax.set_xlim(0, 1)

            if i == 0:
                ax.set_title(model_name)

            if j == 0:
                ax.set_ylabel(f"{dataset}\nAccuracy")

            if i == len(REPRESENTATIVE_MODELS) - 1:
                ax.set_xlabel("Normalized depth")

            ax.grid(alpha=0.3)

    # Common legend
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"module_depth_vision_{filename}"
        save_plot(figname=figname)


def plot_module_depth_vision_all(filename: str, save: bool = False) -> None:
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
    REPRESENTATIVE_DATASETS = df["dataset_name"].unique()
    nrows = len(REPRESENTATIVE_DATASETS)
    ncols = len(REPRESENTATIVE_MODELS)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 3 * nrows),
        # sharex=True,
        # sharey="row",
    )
    for i, dataset in enumerate(REPRESENTATIVE_DATASETS):
        print(dataset)
        for j, model_name in enumerate(REPRESENTATIVE_MODELS):
            print(model_name)
            model_size = REPRESENTATIVE_MODELS[model_name]
            ax = axes[i, j]
            for module in ["act", "rc2"]:
                subset = df[
                    (df["model_name"] == model_name)
                    & (df["model_size"] == model_size)
                    & (df["dataset_name"] == dataset)
                    & (df["module"] == module)
                ].copy()

                if len(subset) == 0:
                    ax.set_axis_off()
                    continue

                subset = subset.sort_values("layer")

                n_layers = subset["layer"].max()

                subset["depth"] = subset["layer"] / n_layers

                ax.plot(subset["depth"], subset["accuracy"], linewidth=2, label=module)
                print(module)
                print(subset["accuracy"].values[-1])
            ax.set_xlim(0, 1)

            if i == 0:
                ax.set_title(model_name)

            if j == 0:
                ax.set_ylabel(f"{dataset}\nAccuracy")

            if i == len(REPRESENTATIVE_MODELS) - 1:
                ax.set_xlabel("Normalized depth")

            ax.grid(alpha=0.3)

    # Common legend
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"module_depth_vision_{filename}"
        save_plot(figname=figname)


def compute_optimal_act_depth(filename: str):
    """
    Compute optimal normalized ACT depth for each model x dataset.

    Requires columns:
    dataset_name, model_name, model_size, layer, module, accuracy
    """

    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep only vision foundation models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # df = df.copy()

    # # number of layers per model variant
    # df["n_layers"] = df.groupby(["model_name", "model_size"])["layer"].transform("max")

    # df["depth"] = df["layer"] / df["n_layers"]

    # # only ACT
    # df_act = df[df["module"] == "act"].copy()

    # results = []

    # for (model_name, model_size, dataset_name), g in df_act.groupby(["model_name", "model_size", "dataset_name"]):
    #     best = g.loc[g["accuracy"].idxmax()]

    #     results.append(
    #         {
    #             "model_name": model_name,
    #             "model_size": model_size,
    #             "dataset_name": dataset_name,
    #             "best_layer": best["layer"],
    #             "n_layers": best["n_layers"],
    #             "best_depth": best["depth"],
    #             "best_accuracy": best["accuracy"],
    #         }
    #     )

    # df = pd.DataFrame(results)

    # model_depth = (
    #     df.groupby(["model_name", "model_size"])
    #     .agg(
    #         median_best_depth=("best_depth", "median"),
    #         mean_best_depth=("best_depth", "mean"),
    #         std_best_depth=("best_depth", "std"),
    #         n_datasets=("dataset_name", "nunique"),
    #     )
    #     .reset_index()
    # )

    # return model_depth

    df = df.copy()
    # number of layers
    df["n_layers"] = df.groupby(["model_name", "model_size"])["layer"].transform("max")

    df["depth"] = df["layer"] / df["n_layers"]

    depth_grid = np.linspace(0, 1, 11)

    results = []

    for (model_name, model_size, dataset_name), g in df.groupby(["model_name", "model_size", "dataset_name"]):
        # baseline = final RC2
        rc2 = g[g["module"] == "rc2"]

        if len(rc2) == 0:
            continue

        baseline = rc2.loc[rc2["layer"].idxmax(), "accuracy"]

        # oracle
        oracle = g["accuracy"].max()

        if oracle <= baseline:
            continue

        # ACT curve
        act = g[g["module"] == "act"].sort_values("depth")

        if len(act) == 0:
            continue

        # recovery at each existing ACT layer
        act_recovery = (act["accuracy"] - baseline) / (oracle - baseline)

        # interpolate to common depth grid
        recovery_interp = np.interp(depth_grid, act["depth"], act_recovery)

        for d, r in zip(depth_grid, recovery_interp):
            results.append(
                {
                    "model": f"{model_name}-{model_size}",
                    "dataset": dataset_name,
                    "depth": d,
                    "recovery": 100 * r,
                }
            )

    results = pd.DataFrame(results)

    # average over datasets
    table = results.groupby(["model", "depth"])["recovery"].mean().unstack()

    # nicer column names
    table.columns = [f"{c:.1f}" for c in table.columns]

    return table.round(1)


def plot_module_depth_vision_(filename: str, save: bool = False) -> None:
    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep only vision foundation models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    DEPTH_GRID = np.linspace(0, 1, 101)

    # ==========================================================
    # Add normalized depth
    # ==========================================================

    df_plot = df.copy()

    # infer number of layers for each model variant
    df_plot["n_layers"] = df_plot.groupby(["model_name", "model_size"])["layer"].transform("max")

    df_plot["depth"] = df_plot["layer"] / df_plot["n_layers"]

    # ==========================================================
    # Compute ACT gain curves
    # ==========================================================

    all_curves = []

    for (model_name, model_size, dataset_name), g_all in df_plot.groupby(["model_name", "model_size", "dataset_name"]):
        # Standard probing = final layer RC2
        rc2 = g_all[g_all["module"] == "rc2"]

        if len(rc2) == 0:
            continue

        baseline = rc2.loc[rc2["layer"].idxmax(), "accuracy"]

        # ACT representations
        act = g_all[g_all["module"] == "act"].copy()

        if len(act) == 0:
            continue

        act = act.sort_values("depth")

        # Gain over standard probing
        act["gain"] = act["accuracy"]

        # interpolate over normalized depth
        gain_curve = np.interp(DEPTH_GRID, act["depth"], act["gain"])

        all_curves.append(gain_curve)

    all_curves = np.stack(all_curves)

    # ==========================================================
    # Aggregate
    # ==========================================================

    median_gain = np.median(all_curves, axis=0)

    q25 = np.percentile(all_curves, 25, axis=0)

    q75 = np.percentile(all_curves, 75, axis=0)

    print(f"Number of model x dataset curves: {len(all_curves)}")

    print(f"Maximum median gain: {median_gain.max():.2f}")

    print(f"Depth of maximum median gain: {DEPTH_GRID[np.argmax(median_gain)]:.2f}")

    # ==========================================================
    # Plot
    # ==========================================================

    plt.figure(figsize=(6, 4))

    plt.plot(DEPTH_GRID, median_gain, linewidth=2, label="ACT")

    plt.fill_between(DEPTH_GRID, q25, q75, alpha=0.25, label="25-75 percentile")

    plt.axhline(0, linewidth=1)

    plt.xlabel("Normalized transformer depth")

    plt.ylabel("Gain over standard probing (accuracy points)")

    plt.xlim(0, 1)

    plt.legend(frameon=False)

    # Common legend
    plt.tight_layout()
    if save:
        figname = f"module_depth_vision_{filename}"
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
def analyze_model_size(filename: str) -> None:
    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    results = []
    for family, sizes in MODEL_SIZES.items():
        sub = df[df["model_name"] == family]

        # Build matrix: rows=datasets, cols=sizes
        pivot = (
            sub.pivot_table(index="dataset_name", columns="model_size", values="gain").reindex(columns=sizes).dropna()
        )

        # --- Friedman test ---
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


def analyze_module(filename: str) -> None:
    # ============================================================

    # Recover results
    path = RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    # ============================================================
    # 2) Recovery tables
    # ============================================================

    CANONICAL_MODULE = "act"
    STANDARD_MODULE = "rc2"

    groups = ["dataset_name", "model_name", "model_size"]

    rows = []

    for key, sub in df.groupby(groups):
        # -----------------------
        # Standard
        # -----------------------
        last_layer = sub["layer"].max()

        std = sub[(sub.layer == last_layer) & (sub.module == STANDARD_MODULE)]["accuracy"]

        if len(std) == 0:
            continue

        std = std.iloc[0]

        # -----------------------
        # Last layer + act
        # -----------------------
        act_last = sub[(sub.layer == last_layer) & (sub.module == CANONICAL_MODULE)]["accuracy"]

        act_last = act_last.iloc[0] if len(act_last) else np.nan

        # -----------------------
        # Best layer + rc2
        # -----------------------
        rc2_best = sub[sub.module == STANDARD_MODULE].sort_values("accuracy").accuracy.iloc[-1]

        # -----------------------
        # Best layer + act
        # -----------------------
        act_best = sub[sub.module == CANONICAL_MODULE].sort_values("accuracy").accuracy.iloc[-1]

        # -----------------------
        # Optimum
        # -----------------------
        optimum = sub["accuracy"].max()

        gain_opt = optimum - std

        if gain_opt <= 1e-8:
            continue

        strategies = {
            "Standard": std,
            "Best layer + rc2": rc2_best,
            "Last layer + act": act_last,
            "Best layer + act": act_best,
        }

        for strategy, acc in strategies.items():
            gain = acc - std

            recovered = np.clip(gain / gain_opt, 0, 1)

            exact = np.isclose(acc, optimum)

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

    # ============================================================
    # Final table
    # ============================================================

    order = [
        "Standard",
        "Last layer + act",
        "Best layer + rc2",
        "Best layer + act",
    ]

    # ------------------------------------------------------------
    # Family table
    # ------------------------------------------------------------
    family = (
        summary.groupby(["family", "strategy"])
        .agg(
            recovery=("recovered", "mean"),
            optimum=("exact", "mean"),
        )
        .reset_index()
    )

    family["optimum"] *= 100

    recovery_family = family.pivot(index="family", columns="strategy", values="recovery").reindex(columns=order)

    optimum_family = family.pivot(index="family", columns="strategy", values="optimum").reindex(columns=order)

    # ------------------------------------------------------------
    # Overall row
    # ------------------------------------------------------------
    overall = summary.groupby("strategy").agg(
        recovery=("recovered", "mean"),
        optimum=("exact", "mean"),
    )

    overall["optimum"] *= 100

    overall_recovery = overall["recovery"].reindex(order).to_frame().T
    overall_recovery.index = ["Overall"]

    overall_optimum = overall["optimum"].reindex(order).to_frame().T
    overall_optimum.index = ["Overall"]

    # ------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------
    recovery = pd.concat([recovery_family, overall_recovery])
    optimum = pd.concat([optimum_family, overall_optimum])

    table = pd.concat(
        {
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


def analyze_model_ranking(filename: str) -> None:
    r"""Display table showing the original model ranking with standard probing and the new ranking
    by optimizing over layers and modules."""
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
def get_tables() -> None:
    # Model size
    filename = "cls"
    analyze_model_size(filename=filename)

    # Module analysis
    filename = "cls"
    analyze_module(filename=filename)


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

    # Module depth
    save = True
    filename = "cls"
    plot_module_depth_vision(filename=filename, save=save)
    # df = compute_optimal_act_depth(filename=filename)
    # print(df.to_latex(index=False, float_format="%.3f"))

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


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire({"table": get_tables, "plot": plot_figures})


# %% CLI
if __name__ == "__main__":
    main()
# %%
