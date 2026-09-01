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
LLM_PALETTE = {"qwen3": "#7987b8", "llama3": "#9fc9b1", "gemma3": "#c6a4e4"}

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
    ax.set_ylabel(r"Percentage of runs with gain $> x$", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    lines, labels = lines[::-1], labels[::-1]
    legend = fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.72, 0.9),
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
        color=GAIN_PALETTE["oracle"],
        edgecolors="#333333",
        linewidths=MARKER_LINEWIDTH,
    )

    # Annotate
    summary_subset = summary.copy()
    summary_subset = summary_subset[summary_subset["dataset_name"].isin(REPRESENTATIVE_DATASETS)]
    for _, row in summary_subset.iterrows():
        if "aircraft" == row["dataset_name"]:
            ax.annotate(
                DATASETS[row["dataset_name"]],
                (row["best_error"], row["mean_gain"]),
                xytext=(-56, 0),
                textcoords="offset points",
                color="#333333",
                fontsize=15,
                va="center",
                ha="left",
            )
        else:
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
    # Regression on all models x dataset points
    df_full = df.merge(summary[["dataset_name", "best_error"]], on="dataset_name")
    rho, pval = spearmanr(df_full["best_error"], df_full["gain"])
    sns.regplot(
        data=df_full,
        x="best_error",
        y="gain",
        color=GAIN_PALETTE["oracle"],
        scatter=False,
        ci=None,
        line_kws={"linewidth": 2, "alpha": 0.8},
        ax=ax,
    )
    handle = Line2D(
        [0],
        [0],
        color=GAIN_PALETTE["oracle"],
        linewidth=2,
        label=rf"$\rho= {rho:.2f}$" + "\n" + rf"$\text{{p-value}}= {pval:.1e}$",
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
    yticks = [0.0, 1.25, 2.5]
    ax.set_yticks(yticks)
    ax.set_xlabel("Corruption severity", fontsize=FONTSIZE)
    ax.set_ylabel(r"Oracle gain $\mathscr{G}_\mathrm{oracle}$ (pp)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        save_plot(figname=f"cifar10c_severity_{filename}")


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

    # Results
    pretraining_labels = {"vit": r"$250$B", "dinov2": r"$1.8$T", "aimv2": r"$3.15$T", "dinov3": r"$3.9$T"}
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
        err_kws={"color": "#333333", "linewidth": 1},
        capsize=0.15,
        ax=ax,
    )

    # Annotate model family name above each bar
    bar_heights = [df[df["model_name"] == m]["gain"].mean() for m in family_order]
    for i, (model_name, height) in enumerate(zip(family_order, bar_heights, strict=True)):
        xytext = (0, 16)
        ax.annotate(
            VFM_NAMES[model_name],
            xy=(i, height),
            xytext=xytext,
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=15,
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
    ax.set_ylabel(r"Oracle gain $\mathscr{G}_\mathrm{oracle}$ (pp)", fontsize=FONTSIZE)
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
        bbox_to_anchor=(0.91, 0.78),
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
    width = 5
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
    width = 5.5
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
                ax.set_ylabel(f"{DATASETS[dataset_name]}\n" + r"Accuracy ($\%$)", fontsize=FONTSIZE)
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
    width = 5.5
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
        err_kws={"color": "#333333", "linewidth": 1},
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
    ax.set_xticklabels(["Best RC2", "Fixed", "Best Act"])
    ax.set_ylim(20, 100)
    ax.set_yticks([26, 46, 66, 86])
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
    height = 5
    ncols = len(LLM_FAMILIES)
    figsize = ncols * width, height
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize, sharey=True)

    # Recover results
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

            # Mark the highest ln2 performance
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
        ax.set_title(model_labels[model_family], fontsize=FONTSIZE)
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

                # Mark the highest ln2 performance
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
    width = 5.5
    height = 5
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
        err_kws={"color": "#333333", "linewidth": 1},
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
    ax.set_xticklabels(["Best RC2", "Fixed", "Best LN2"])
    ax.set_ylim(20, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xlabel("")
    ax.set_ylabel(r"Oracle recovery ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        save_plot(figname=f"oracle_recovery_language_{filename}")


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
    table["Average"] = table.mean(axis=1)
    ordered_models = [m for models in LLM_FAMILIES.values() for m in models]
    ordered_models = [m for m in ordered_models if m in table.columns]
    table = table[ordered_models + ["Average"]]
    print(table.to_latex(float_format="%.2f", bold_rows=True, escape=False))


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
    print(f"There are {len(df)} LLMs runs.")


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

    # Oracle recovery
    save = True
    filename = "cls"
    plot_oracle_recovery_vision(filename=filename, save=save)

    # r"""LLMs plots."""
    # # Model family
    # save = True
    # filename = "mteb"
    # plot_model_family_language(filename=filename, save=save)

    # # Probing depth
    # save = True
    # filename = "mteb"
    # plot_probing_depth_language(filename=filename, save=save)
    # plot_probing_depth_all_language(filename=filename, save=save)

    # Oracle recovery
    save = True
    filename = "mteb"
    plot_oracle_recovery_language(filename=filename, save=save)


def get_tables() -> None:
    r"""VFMs tables."""
    # Model size
    filename = "cls"
    table_model_size_vision(filename=filename)

    # Winning module frequency
    filename = "cls"
    table_winning_module_frequency_vision(filename=filename)

    # Performance comparions
    filename = "cls"
    table_performance_comparison_vision(filename=filename)
    table_performance_comparison_all_vision(filename=filename)

    r"""LLMs tables."""
    # Performance comparions
    filename = "mteb"
    table_performance_comparison_language(filename=filename)

    r"""Number of runs"""
    table_runs()


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
