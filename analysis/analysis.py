r"""
Analyze evaluation results from VFMs and LLMs.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
from typing import Any, cast

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from scipy.stats import friedmanchisquare, spearmanr

from waldo.infra import FIGURE_DIR, RESULT_DIR

logger = logging.getLogger("waldo.analysis")

# Paths
RUN_RESULT_DIR = RESULT_DIR / "runs"
ABLATION_RESULT_DIR = RESULT_DIR / "ablations"

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
    "eurosat": "EuroSAT",
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
# Models of the paper, selected by name (e.g., Qwen3-Base models are not part of the paper)
LLMs = [
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Llama-3.2-1B",
    "Llama-3.2-3B",
    "Llama-3.1-8B",
    "gemma-3-1b-pt",
    "gemma-3-4b-pt",
    "gemma-3-12b-pt",
]
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
TITLE_FONTSIZE = FONTSIZE + 5
LEG_FONTSIZE = FONTSIZE
LINEWIDTH = 4
BARWIDTH = 0.25
MARKER_LINEWIDTH = 0.5
ALPHA_GRID = 0.8
ZOOM_SCALE = 0.7
GAIN_PALETTE = {
    "layer": "#7987B8",
    "module": "#37abb5",
    "oracle": "#c6a4e4",
}
MODULE_PALETTE = {
    "ln1": "#daa4ac",
    "mha": "#4997d0",
    "rc1": "#fffdd0",
    "ln2": "#b153a1",
    "fc1": "#a291e1",
    "act": "#96c0cf",
    "fc2": "#858ec2",
    "rc2": "#848482",
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
    path = RUN_RESULT_DIR

    # Vision
    comp = 0
    for filename in ["cls", "cifar10_c"]:
        df = pd.read_csv(path / f"{filename}.csv")
        df = df.copy()
        pattern = "|".join(VFMs)
        df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])
        print(f"There are {len(df)} runs for {filename}.")
        comp += len(df)
    print(f"There are {comp} VFMs runs.")

    # Language
    filename = "mteb"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()
    df = cast(pd.DataFrame, df[df["model_name"].isin(LLMs)])
    comp += len(df)
    print(f"There are {len(df)} LLMs runs.")

    # Total
    print(f"There are a total of {comp} runs.")


def get_ordered_dataset(filename: str) -> list:
    # Recover results
    path = RUN_RESULT_DIR / "summary"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = cast(pd.DataFrame, df[df["dataset_name"].str.contains(pattern, case=False, na=False)])
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Recover mean probing error across models
    ordered_dataset = df.groupby("dataset_name")["baseline_accuracy"].mean().rsub(100).sort_values().index.tolist()

    return ordered_dataset


# ------------------------------------------------------------------------------
# Intro
# ------------------------------------------------------------------------------
def plot_intro(vision_filename: str = "cls", language_filename: str = "mteb", save: bool = False) -> None:
    r"""
    Plot side-by-side optimal layer-module heatmaps for VFMs and LLMs.
    """
    # Colorbar
    COLORBAR_MIN = 0
    COLORBAR_MAX = 20
    cmap = sns.cubehelix_palette(rot=0.2, dark=0.45, light=0.95, as_cmap=True)
    norm = PowerNorm(gamma=0.6, vmin=COLORBAR_MIN, vmax=COLORBAR_MAX)
    bar_color = cmap(norm(COLORBAR_MAX))
    star_color = "tab:red"
    circle_color = "#333333"

    # Figure
    depth_step = 9
    n_bins = int(round(100 / depth_step))
    n_modules = 8
    cell_width = 0.5
    cell_height = 0.4
    heat_width = cell_width * n_bins
    heat_height = cell_height * n_modules
    top_height, right_width, gap_width, cbar_width = 1.1, 1.3, 1.0, 0.2
    width_ratios = [heat_width, right_width, gap_width, heat_width, right_width, 0.3, cbar_width]
    height_ratios = [top_height, heat_height]
    width = sum(width_ratios) / 0.9
    height = sum(height_ratios) / 0.78
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(2, 7, figure=fig, width_ratios=width_ratios, height_ratios=height_ratios, wspace=0.0, hspace=0.0)

    # Variables
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
    panel_titles = ["Vision Foundation Models", "Large Language Models"]
    bin_width = 100 / n_bins
    depth_edges = np.linspace(0, 100, n_bins + 1)
    module_centers = np.arange(len(modules)) + 0.5
    depth_ticks = [0, 50, 100]
    module_to_idx = {module: i for i, module in enumerate(modules)}
    strategy_modules = ["act", "ln2"]
    strategy_depths = [1.0, 0.60]
    baseline_module = "rc2"
    depth_grid = np.linspace(0, 100, 501)
    depth_sigma = bin_width / 2

    # Recover results
    pair_percentages = []
    module_percentages = []
    depth_percentages = []
    depth_densities = []
    smooth_percentages = []
    strategy_bins = []
    for modality_idx, (modality, filename) in enumerate(
        zip(("vision", "language"), (vision_filename, language_filename), strict=True)
    ):
        path = RUN_RESULT_DIR / "summary"
        df = pd.read_csv(path / f"{filename}.csv")
        df = df.copy()
        if modality == "vision":
            pattern = "|".join(VFMs)
            df = cast(
                pd.DataFrame,
                df[df["model_name"].str.contains(pattern, case=False, na=False)],
            )
            pattern = "|".join(DATASETS.keys())
            df = cast(
                pd.DataFrame,
                df[df["dataset_name"].str.contains(pattern, case=False, na=False)],
            )
        else:
            df = cast(pd.DataFrame, df[df["model_name"].isin(LLMs)])

        logger.info(f"{modality}: results are presented over a total of {len(df)} runs.")

        # Optimal layer-module pairs
        best_depth = 100 * df["best_layer"].to_numpy(dtype=float) / (df["n_layers"].to_numpy(dtype=float) - 1)
        bin_index = np.minimum((best_depth / bin_width).astype(int), len(depth_edges) - 2)
        pair_counts = np.zeros((len(depth_edges) - 1, len(modules)), dtype=float)
        for depth_idx, module in zip(bin_index, df["best_module"].to_numpy(dtype=str), strict=True):
            module_idx = module_to_idx.get(module)
            if module_idx is not None:
                pair_counts[depth_idx, module_idx] += 1

        pair_pct = 100 * pair_counts / len(df)
        pair_percentages.append(pair_pct)
        module_percentages.append(pair_pct.sum(axis=0))
        depth_percentages.append(pair_pct.sum(axis=1))

        # Continuous depth marginal
        depth_density = sum(
            np.exp(-0.5 * ((depth_grid - center) / depth_sigma) ** 2)
            for depth in best_depth
            for center in (depth, -depth, 200 - depth)
        )
        depth_densities.append(depth_density / len(df))

        # Heatmap smoothed along depth
        smooth_pct = np.zeros((len(modules), len(depth_grid)))

        for depth, module in zip(best_depth, df["best_module"].to_numpy(dtype=str), strict=True):
            kernel = sum(
                np.exp(-0.5 * ((depth_grid - center) / depth_sigma) ** 2) for center in (depth, -depth, 200 - depth)
            )
            kernel /= depth_sigma * np.sqrt(2 * np.pi)
            smooth_pct[module_to_idx[module]] += 100 * bin_width * kernel / len(df)
        smooth_percentages.append(smooth_pct)

        # Fixed strategy depth
        n_layers = df["n_layers"].to_numpy(dtype=int)
        strategy_layer = (strategy_depths[modality_idx] * (n_layers - 1)).astype(int)
        strategy_depth = 100 * strategy_layer / (n_layers - 1)
        strategy_bin = np.minimum((strategy_depth / bin_width).astype(int), len(depth_edges) - 2)
        strategy_bins.append(int(np.bincount(strategy_bin).argmax()))

    # Visualization
    reference_heatmap = None
    im = None
    for panel_idx, title in enumerate(panel_titles):
        col = 0 if panel_idx == 0 else 3
        ax_top = fig.add_subplot(gs[0, col])
        if reference_heatmap is None:
            ax_heat = fig.add_subplot(gs[1, col])
            reference_heatmap = ax_heat
        else:
            ax_heat = fig.add_subplot(gs[1, col], sharey=reference_heatmap)
        ax_right = fig.add_subplot(gs[1, col + 1])
        im = ax_heat.imshow(
            smooth_percentages[panel_idx],
            extent=(0, len(depth_edges) - 1, len(modules), 0),
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        for row in range(1, len(modules)):
            ax_heat.axhline(row, color="white", linewidth=1.5)
        depth_x = depth_grid / bin_width
        ax_top.fill_between(depth_x, 0, depth_densities[panel_idx], color=bar_color, alpha=0.35, linewidth=0)
        ax_top.plot(depth_x, depth_densities[panel_idx], color=bar_color, linewidth=2)

        # Discrete module marginal
        module_pct = pair_percentages[panel_idx]
        ax_right.barh(
            module_centers,
            module_pct.sum(axis=0),
            height=0.75,
            color=bar_color,
            alpha=0.35,
            edgecolor=bar_color,
            linewidth=1,
        )

        # Standard probing and fixed strategy
        baseline_x = len(depth_edges) - 1
        strategy_x = 100 * strategy_depths[panel_idx] / bin_width
        ax_heat.scatter(
            [baseline_x],
            [module_centers[module_to_idx[baseline_module]]],
            s=220,
            color="white",
            edgecolor=circle_color,
            linewidth=2,
            zorder=5,
            clip_on=False,
        )
        ax_heat.scatter(
            [strategy_x],
            [module_centers[module_to_idx[strategy_modules[panel_idx]]]],
            s=500,
            marker=MarkerStyle("*"),
            color=star_color,
            edgecolor=circle_color,
            linewidth=0.5,
            zorder=6,
            clip_on=False,
        )

        # Heatmap
        x_max = len(depth_edges) - 1
        ax_heat.set(
            xlim=(0, x_max),
            ylim=(len(modules), 0),
            xticks=[tick / bin_width for tick in depth_ticks],
            xticklabels=[str(tick) for tick in depth_ticks],
            yticks=module_centers,
            yticklabels=[module_labels[module] for module in modules],
        )

        # Visualization
        ax_heat.spines["left"].set_linewidth(1)
        ax_heat.spines["right"].set_linewidth(1)
        ax_heat.spines["top"].set_linewidth(1)
        ax_heat.spines["bottom"].set_linewidth(1)
        ax_heat.tick_params(axis="both", direction="out", length=5, width=1)
        sns.despine(fig, ax_heat, trim=True, right=True, offset=10)
        ax_heat.set_xlabel("Layer depth (%)", fontsize=FONTSIZE)
        ax_heat.text(
            0.5,
            -0.40,
            title,
            transform=ax_heat.transAxes,
            ha="center",
            va="top",
            fontsize=TITLE_FONTSIZE,
        )
        # Top marginal
        ax_top.set_xlim(0, x_max)
        ax_top.set_ylim(0, 1.1 * depth_densities[panel_idx].max())
        ax_top.set_xticks([])

        # Right marginal
        module_max = module_pct.sum(axis=0).max()
        ax_right.set_ylim(len(modules), 0)
        ax_right.set_xlim(-0.2 * module_max, 1.1 * module_max)
        ax_right.set_xticks([])
        ax_right.set_yticks([])

        for axis in (ax_top, ax_right):
            for spine in axis.spines.values():
                spine.set_visible(False)

        ax_top.set_axis_off()
        ax_right.set_axis_off()

    if im is not None:
        ax_cbar = fig.add_subplot(gs[:, 6])
        cbar_ticks = np.linspace(COLORBAR_MIN, COLORBAR_MAX, 4, dtype=int)
        cbar = fig.colorbar(im, cax=ax_cbar, ticks=cbar_ticks)
        cbar.ax.set_yticklabels([str(tick) for tick in cbar_ticks])
        cbar.set_label("Optimality frequency (%)", fontsize=FONTSIZE, rotation=270, labelpad=25)
        cbar.ax.tick_params(labelsize=FONTSIZE - 5)
        fig.set_layout_engine("none")  # pyright: ignore[reportArgumentType]
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.92)

    if save:
        figname = "intro"
        save_plot(figname=figname)


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
    path = RUN_RESULT_DIR / "summary"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = cast(pd.DataFrame, df[df["dataset_name"].str.contains(pattern, case=False, na=False)])
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
    ax.set_xlabel("Gain over probing baseline (pp)", fontsize=FONTSIZE)
    ax.set_ylabel(r"Runs with gain $> x$ ($\%$)", fontsize=FONTSIZE)
    x, y = ax.yaxis.label.get_position()
    ax.yaxis.label.set_position((x, 0.4))
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
    path = RUN_RESULT_DIR / "summary"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = cast(pd.DataFrame, df[df["dataset_name"].str.contains(pattern, case=False, na=False)])
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
    container = ax.errorbar(
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
            str(DATASETS[str(row["dataset_name"])]),
            (float(row["transfer_error"]), float(row["mean_gain"])),
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
    path = RUN_RESULT_DIR / "summary"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])

    # Corruption severity
    severity_order = [1, 3, 5]
    sns.barplot(
        data=df,
        x="severity",
        y="gain",
        order=severity_order,
        estimator=np.mean,
        errorbar=("ci", 95),
        seed=0,
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

    path = RUN_RESULT_DIR / "summary"
    df = pd.read_csv(path / f"{filename}.csv")
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)].copy())
    df["depth"] = 100 * df["best_layer"] / (df["n_layers"] - 1)

    severity_order = [1, 3, 5]
    sns.barplot(
        data=df,
        x="severity",
        y="depth",
        order=severity_order,
        estimator=np.mean,
        errorbar=("ci", 95),
        seed=0,
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
    path = RUN_RESULT_DIR / "summary"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = cast(pd.DataFrame, df[df["dataset_name"].str.contains(pattern, case=False, na=False)])
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Compute error reduction and relative gain
    df["error_reduction"] = 100 * df["gain"] / (100 - df["baseline_accuracy"])
    print(f"error reduction: {df.groupby('model_name')['error_reduction'].mean()}")

    # Results
    pretraining_labels = {"vit": r"$0.25$T", "dinov2": r"$1.75$T", "aimv2": r"$3.15$T", "dinov3": r"$3.90$T"}
    family_order = ["vit", "dinov2", "aimv2", "dinov3"]
    sns.barplot(
        data=df,
        x="model_name",
        y="gain",
        order=family_order,
        estimator=np.mean,
        errorbar=("ci", 95),
        seed=0,
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
            xy=(float(i), float(height)),
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
    height = 5
    ncols = len(REPRESENTATIVE_DATASETS)
    figsize = ncols * width, height
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize, sharey=True)

    # Recover results
    path = RUN_RESULT_DIR / "summary"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = cast(pd.DataFrame, df[df["dataset_name"].str.contains(pattern, case=False, na=False)])
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_VFMs.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = cast(pd.DataFrame, df[mask])

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
        gain_map = dict(zip(gain_types, gain_labels, strict=False))
        agg_long["gain_type"] = agg_long["gain_type"].map(gain_map)
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
            ax.set_ylabel("Gain over probing baseline (pp)", fontsize=FONTSIZE)
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
    path = RUN_RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = cast(pd.DataFrame, df[df["dataset_name"].str.contains(pattern, case=False, na=False)])

    # Recover representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_VFMs.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = cast(pd.DataFrame, df[mask])
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
                    subset = cast(pd.DataFrame, subset).sort_values(by="layer")
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
    # Recover dataset order
    ordered_dataset = get_ordered_dataset(filename=filename)

    # Figure
    width = 5
    height = 4
    sizes = VFM_SIZES[model_name]
    ncols = len(sizes)

    # Recover results
    path = RUN_RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join([model_name])
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = cast(pd.DataFrame, df[df["dataset_name"].str.contains(pattern, case=False, na=False)])
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    model_labels = {
        "vit": ["ViT-B", "ViT-L", "ViT-H"],
        "dinov2": ["DINOv2-S", "DINOv2-B", "DINOv2-L", "DINOv2-g"],
        "dinov3": ["DINOv3-S", "DINOv3-B", "DINOv3-L", "DINOv3-7B"],
        "aimv2": ["AIMv2-L", "AIMv2-H", "AIMv2-1B", "AIMv2-3B"],
    }

    n_parts = 3
    nrows = len(ordered_dataset) // n_parts
    for part in range(n_parts):
        figsize = ncols * width, nrows * height
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
        datasets = ordered_dataset[nrows * part : nrows * (part + 1)]
        for i, dataset_name in enumerate(datasets):
            for j, model_size in enumerate(sizes):
                ax = axes[i, j]
                for module in ["act", "rc2"]:
                    subset = df[
                        (df["model_name"] == model_name)
                        & (df["model_size"] == model_size)
                        & (df["dataset_name"] == dataset_name)
                        & (df["module"] == module)
                    ].copy()
                    subset = cast(pd.DataFrame, subset).sort_values(by="layer")
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
                    ax.set_title(f"{model_labels[model_name][j]}\n", fontsize=FONTSIZE)
                if i == nrows - 1:
                    ax.set_xlabel(r"Layer depth ($\%$)", fontsize=FONTSIZE)
                if j == 0:
                    ax.set_ylabel(f"{DATASETS[dataset_name]}\n" + r"Accuracy ($\%$)", fontsize=FONTSIZE)
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
            figname = f"probing_depth_family_{model_name}_part_{part}_vision_{filename}"
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
    width = 5
    height = 4
    figsize = width, height
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    # Recover results
    path = RUN_RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = cast(pd.DataFrame, df[df["dataset_name"].str.contains(pattern, case=False, na=False)])
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    order = ["Best RC2", "Fixed", "Best ACT"]
    groups = ["dataset_name", "model_name", "model_size"]
    rows = []
    for key, sub in cast(Any, df.groupby(groups)):
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
                        "dataset": str(key[0]),
                        "family": str(key[1]),
                        "size": str(key[2]),
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
        seed=0,
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
    ax.set_xticklabels([r"Best $\mathtt{RC2}$", r"$\mathtt{Act@100}$", r"Best $\mathtt{Act}$"])
    ax.set_ylim(15, 85)
    ax.set_yticks([15, 50, 85])
    ax.set_xlabel("")
    ax.set_ylabel(r"Oracle gain recovered ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        save_plot(figname=f"oracle_recovery_vision_{filename}")


def plot_probing_depth_clip_vision(save: bool = False) -> None:
    r"""
    Evolution of the act and rc2 probing accuracy along with the depth for CLIP-L.

    Parameters
    ----------
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 5
    height = 4
    ncols = len(REPRESENTATIVE_DATASETS)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * width, height), sharex=True)

    # Recover results
    df = pd.read_csv(ABLATION_RESULT_DIR / "cls.csv")
    df = df.loc[df["ablation"] == "clip"]
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    for j, dataset_name in enumerate(REPRESENTATIVE_DATASETS):
        ax = axes[j]
        for module in ["rc2", "act"]:
            subset = df.loc[(df["dataset_name"] == dataset_name) & (df["module"] == module)].copy()
            subset = subset.sort_values(by="layer")
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
        ax.set_xlabel(r"Layer depth ($\%$)", fontsize=FONTSIZE)
        ax.set_title(f"{DATASETS[dataset_name]}\n", fontsize=FONTSIZE)
        if j == 0:
            ax.set_ylabel("CLIP\n" + r"Accuracy ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, axes[0], trim=True, right=True, offset=10)

    # Common legend
    legend = axes[0].legend(
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
        figname = "probing_depth_clip_vision"
        save_plot(figname=figname, dirname="ablations")


# ------------------------------------------------------------------------------
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
    path = RUN_RESULT_DIR / "summary"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = cast(pd.DataFrame, df[df["dataset_name"].str.contains(pattern, case=False, na=False)])
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
    path = RUN_RESULT_DIR / "summary"
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep only vision foundation models
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])

    # Keep datasets
    pattern = "|".join(DATASETS.keys())
    df = cast(pd.DataFrame, df[df["dataset_name"].str.contains(pattern, case=False, na=False)])
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    freq = df["best_module"].value_counts(normalize=True).mul(100)
    freq = freq.reindex(MODULE_ORDER, fill_value=0.0)

    table = freq.to_frame().T
    table.index = ["Winning freq. (%)"]
    print(table.to_latex(index=True, float_format="%.1f"))


def table_performance_comparison_vision(filename: str) -> None:
    r"""Get results on large vision models, with datasets ordered by increasing transfer difficulty."""
    # Recover dataset ordered by increasing transfer difficutly
    ordered_dataset = get_ordered_dataset(filename=filename)

    # Recover results
    path = RUN_RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])

    # Keep datasets
    pattern = "|".join(DATASETS)
    df = cast(pd.DataFrame, df[df["dataset_name"].str.contains(pattern, case=False, na=False)])

    # Keep representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_VFMs.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = cast(pd.DataFrame, df[mask])
    family_order: list[str] = list(REPRESENTATIVE_VFMs.keys())

    # Results
    last_layer = df.groupby(["dataset_name", "model_name"])["layer"].transform("max")
    model_labels = {"vit": "ViT", "dinov2": "DINOv2", "dinov3": "DINOv3", "aimv2": "AIMv2"}
    dataset_labels = {
        "cifar10": "Cifar10",
        "cifar100": "Cifar100",
        "dtd": "DTD",
        "eurosat": "EuroSAT",
        "flowers102": "Flowers102",
        "food101": "Food-101",
        "pet": "Pet",
        "cifar10_c-corruption-contrast-severity-5": "Contrast",
        "cifar10_c-corruption-motion_blur-severity-5": "Motion Blur",
        "cifar10_c-corruption-speckle_noise-severity-5": "Speckle Noise",
        "aircraft": "Aircraft",
        "domainnet-clipart": "Clipart",
        "domainnet-infograph": "Infograph",
        "domainnet-quickdraw": "Quickdraw",
        "domainnet-painting": "Painting",
        "domainnet-sketch": "Sketch",
        "iwildcam": "iWildCam",
        "rxrx1": "RxRx1",
    }
    rows: list[dict[str, object]] = []
    for key_values, sub in cast(Any, df.groupby(["dataset_name", "model_name"])):
        dataset, family = key_values
        last_layer = int(sub["layer"].max())
        standard = float(
            cast(pd.Series, sub[(sub["layer"] == last_layer) & (sub["module"] == "rc2")]["accuracy"]).iloc[0]
        )
        fixed = float(cast(pd.Series, sub[(sub["layer"] == last_layer) & (sub["module"] == "act")]["accuracy"]).iloc[0])
        oracle = float(sub["accuracy"].max())
        rows.append(
            {
                "dataset": str(dataset),
                "family": str(family),
                "standard": standard,
                "fixed": fixed - standard,
                "oracle": oracle - standard,
            }
        )
    summary: pd.DataFrame = pd.DataFrame(rows)

    def format_gain(value: float) -> str:
        if round(value, 1) > 0:
            return rf"\add{{+{value:.1f}}}"
        if round(value, 1) < 0:
            return rf"\sub{{{value:.1f}}}"
        return "0.0"

    def format_accuracy(value: float) -> str:
        return f"{value:.1f}"

    header = (
        ["Strategy"]
        + [rf"\rotatebox{{90}}{{{dataset_labels.get(dataset, dataset)}}}" for dataset in ordered_dataset]
        + [r"\rotatebox{90}{Average}"]
    )
    n_result_columns = len(ordered_dataset) + 1
    column_spec = "l" + "c" * n_result_columns
    lines = [rf"\begin{{NiceTabular}}{{{column_spec}}}", r"\toprule", " & ".join(header), r"\\[-1pt]", r"\midrule"]
    n_columns = n_result_columns + 1
    for family in family_order:
        family_df = cast(pd.DataFrame, summary[summary["family"] == family])
        lines.append(rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{{model_labels[family]}}}}} \\")
        lines.extend([r"\arrayrulecolor{gray!50}", r"\midrule"])
        values = [
            format_accuracy(float(family_df.loc[family_df["dataset"] == str(dataset), "standard"].to_numpy()[0]))
            for dataset in ordered_dataset
        ]
        values.append(format_accuracy(float(family_df["standard"].mean())))
        lines.append(r"\texttt{Baseline} & " + " & ".join(values) + r" \\")
        values = [
            format_gain(float(family_df.loc[family_df["dataset"] == str(dataset), "fixed"].to_numpy()[0]))
            for dataset in ordered_dataset
        ]
        values.append(format_gain(float(family_df["fixed"].mean())))
        lines.append(r"$\vfm$ & " + " & ".join(values) + r" \\")
        values = [
            format_gain(float(family_df.loc[family_df["dataset"] == str(dataset), "oracle"].to_numpy()[0]))
            for dataset in ordered_dataset
        ]
        values.append(format_gain(float(family_df["oracle"].mean())))
        lines.append(r"\oraclerow \texttt{Oracle} & " + " & ".join(values) + r" \\")
        lines.append(r"\arrayrulecolor{black}")
        if family != family_order[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{NiceTabular}"])
    print("\n".join(lines))


def table_performance_comparison_all_vision(filename: str) -> None:
    r"""Get results on all vision models, with datasets ordered by increasing transfer difficulty."""
    # Recover dataset ordered by increasing transfer difficutly
    ordered_dataset = get_ordered_dataset(filename=filename)

    # Recover results
    path = RUN_RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])

    # Keep datasets
    pattern = "|".join(DATASETS)
    df = cast(pd.DataFrame, df[df["dataset_name"].str.contains(pattern, case=False, na=False)])

    # Results
    last_layer = df.groupby(["dataset_name", "model_name", "model_size"])["layer"].transform("max")
    family_order: list[str] = list(VFM_NAMES.keys())
    model_labels = {
        "vit": ["B", "L", "H"],
        "dinov2": ["S", "B", "L", "g"],
        "dinov3": ["S", "B", "L", "7B"],
        "aimv2": ["L", "H", "1B", "3B"],
    }
    dataset_labels = {
        "cifar10": "Cifar10",
        "cifar100": "Cifar100",
        "dtd": "DTD",
        "eurosat": "EuroSAT",
        "flowers102": "Flowers102",
        "food101": "Food-101",
        "pet": "Pet",
        "cifar10_c-corruption-contrast-severity-5": "Contrast",
        "cifar10_c-corruption-motion_blur-severity-5": "Motion Blur",
        "cifar10_c-corruption-speckle_noise-severity-5": "Speckle Noise",
        "aircraft": "Aircraft",
        "domainnet-clipart": "Clipart",
        "domainnet-infograph": "Infograph",
        "domainnet-quickdraw": "Quickdraw",
        "domainnet-painting": "Painting",
        "domainnet-sketch": "Sketch",
        "iwildcam": "iWildCam",
        "rxrx1": "RxRx1",
    }
    rows: list[dict[str, object]] = []
    for key_values, sub in cast(Any, df.groupby(["dataset_name", "model_name", "model_size"])):
        dataset, family, size = key_values
        last_layer = int(sub["layer"].max())
        standard = float(
            cast(pd.Series, sub[(sub["layer"] == last_layer) & (sub["module"] == "rc2")]["accuracy"]).iloc[0]
        )
        fixed = float(cast(pd.Series, sub[(sub["layer"] == last_layer) & (sub["module"] == "act")]["accuracy"]).iloc[0])
        oracle = float(sub["accuracy"].max())
        rows.append(
            {
                "dataset": str(dataset),
                "family": str(family),
                "size": str(size),
                "standard": standard,
                "fixed": fixed,
                "oracle": oracle,
            }
        )
    summary: pd.DataFrame = pd.DataFrame(rows)
    header = (
        ["Model", "Strategy"]
        + [rf"\rotatebox{{90}}{{{dataset_labels.get(dataset, dataset)}}}" for dataset in ordered_dataset]
        + [r"\rotatebox{90}{Average}"]
    )
    n_columns = len(ordered_dataset) + 3
    lines = [
        rf"\begin{{NiceTabular}}{{{'l' + 'c' * (n_columns - 1)}}}",
        r"\toprule",
        " & ".join(header),
        r"\\[-1pt]",
        r"\midrule",
    ]
    for family_idx, family in enumerate(family_order):
        family_df = cast(pd.DataFrame, summary[summary["family"] == family])
        lines.append(rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{{VFM_NAMES[family]}}}}} \\")
        lines.extend([r"\arrayrulecolor{gray!50}", r"\midrule"])
        for model_idx, model_label in enumerate(model_labels[family]):
            model_size = VFM_SIZES[family][model_idx]
            model_rows = family_df[family_df["size"].eq(model_size)]
            if model_rows.empty:
                continue
            values = [
                f"{float(model_rows.loc[model_rows['dataset'] == str(dataset), 'standard'].to_numpy()[0]):.1f}"
                for dataset in ordered_dataset
            ]
            values.append(f"{float(model_rows['standard'].mean()):.1f}")
            lines.append(
                rf"\multirow{{3}}{{*}}{{{model_label}}} & \texttt{{Baseline}} & " + " & ".join(values) + r" \\"
            )
            values = [
                f"{float(model_rows.loc[model_rows['dataset'] == str(dataset), 'fixed'].to_numpy()[0]):.1f}"
                for dataset in ordered_dataset
            ]
            values.append(f"{float(model_rows['fixed'].mean()):.1f}")
            lines.append(r" & $\vfm$ & " + " & ".join(values) + r" \\")
            values = [
                f"{float(model_rows.loc[model_rows['dataset'] == str(dataset), 'oracle'].to_numpy()[0]):.1f}"
                for dataset in ordered_dataset
            ]
            values.append(f"{float(model_rows['oracle'].mean()):.1f}")
            lines.append(r" & \oraclerow \texttt{Oracle} & " + " & ".join(values) + r" \\")
            if model_idx != len(model_labels[family]) - 1:
                lines.extend([r"\arrayrulecolor{gray!50}", rf"\cline{{2-{n_columns}}}"])
        lines.append(r"\arrayrulecolor{black}")
        if family_idx != len(family_order) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{NiceTabular}"])
    print("\n".join(lines))


def table_performance_comparison_act_fc1_vision(filename: str) -> None:
    r"""Compare final-layer RC2, ACT, and FC1 on large vision models."""
    # Recover dataset ordered by increasing transfer difficulty
    ordered_dataset = get_ordered_dataset(filename=filename)

    # Recover results
    path = RUN_RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])

    # Keep datasets
    pattern = "|".join(DATASETS)
    df = cast(pd.DataFrame, df[df["dataset_name"].str.contains(pattern, case=False, na=False)])

    # Keep representative models
    mask = np.zeros(len(df), dtype=bool)
    for model, size in REPRESENTATIVE_VFMs.items():
        mask |= (df["model_name"] == model) & (df["model_size"] == size)
    df = cast(pd.DataFrame, df[mask])
    family_order: list[str] = list(REPRESENTATIVE_VFMs.keys())

    # Results
    model_labels = {"vit": "ViT", "dinov2": "DINOv2", "dinov3": "DINOv3", "aimv2": "AIMv2"}
    dataset_labels = {
        "cifar10": "Cifar10",
        "cifar100": "Cifar100",
        "dtd": "DTD",
        "eurosat": "EuroSAT",
        "flowers102": "Flowers102",
        "food101": "Food-101",
        "pet": "Pet",
        "cifar10_c-corruption-contrast-severity-5": "Contrast",
        "cifar10_c-corruption-motion_blur-severity-5": "Motion Blur",
        "cifar10_c-corruption-speckle_noise-severity-5": "Speckle Noise",
        "aircraft": "Aircraft",
        "domainnet-clipart": "Clipart",
        "domainnet-infograph": "Infograph",
        "domainnet-quickdraw": "Quickdraw",
        "domainnet-painting": "Painting",
        "domainnet-sketch": "Sketch",
        "iwildcam": "iWildCam",
        "rxrx1": "RxRx1",
    }

    rows: list[dict[str, object]] = []
    for key_values, sub in cast(Any, df.groupby(["dataset_name", "model_name"])):
        dataset, family = key_values
        last_layer = int(sub["layer"].max())

        def get_accuracy(module: str, sub: pd.DataFrame = sub, last_layer: int = last_layer) -> float:
            return float(
                cast(
                    pd.Series,
                    sub[(sub["layer"] == last_layer) & (sub["module"] == module)]["accuracy"],
                ).iloc[0]
            )

        rows.append(
            {
                "dataset": str(dataset),
                "family": str(family),
                "standard": get_accuracy(module="rc2"),
                "fc1": get_accuracy(module="fc1"),
                "act": get_accuracy(module="act"),
            }
        )

    summary: pd.DataFrame = pd.DataFrame(rows)

    def format_accuracy(value: float) -> str:
        return f"{value:.1f}"

    header = (
        ["Strategy"]
        + [rf"\rotatebox{{90}}{{{dataset_labels.get(dataset, dataset)}}}" for dataset in ordered_dataset]
        + [r"\rotatebox{90}{Average}"]
    )
    n_result_columns = len(ordered_dataset) + 1
    column_spec = "l" + "c" * n_result_columns
    lines = [
        rf"\begin{{NiceTabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(header),
        r"\\[-1pt]",
        r"\midrule",
    ]
    n_columns = n_result_columns + 1

    for family_idx, family in enumerate(family_order):
        family_df = cast(pd.DataFrame, summary[summary["family"] == family])
        lines.append(rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{{model_labels[family]}}}}} \\")
        lines.extend([r"\arrayrulecolor{gray!50}", r"\midrule"])

        for column, label in [
            ("standard", r"\texttt{Baseline}"),
            ("fc1", r"$\texttt{FC1@100}$"),
            ("act", r"$\texttt{Act@100}$"),
        ]:
            values = [
                format_accuracy(float(family_df.loc[family_df["dataset"] == str(dataset), column].to_numpy()[0]))
                for dataset in ordered_dataset
            ]
            values.append(format_accuracy(float(family_df[column].mean())))
            lines.append(label + " & " + " & ".join(values) + r" \\")

        lines.append(r"\arrayrulecolor{black}")
        if family_idx != len(family_order) - 1:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{NiceTabular}"])
    print("\n".join(lines))


def table_global_statistics_vision(filename: str) -> None:
    r"""
    Compute win rates for fixed and depth-optimized vision strategies.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    """
    path = RUN_RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep vision models and datasets
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])
    pattern = "|".join(DATASETS.keys())
    df = cast(pd.DataFrame, df[df["dataset_name"].str.contains(pattern, case=False, na=False)])

    def print_statistics(frame: pd.DataFrame, scope: str) -> None:
        rows = []
        for _, sub in frame.groupby(["dataset_name", "model_name", "model_size"]):
            last_layer = sub["layer"].max()
            rows.append(
                {
                    "standard": float(
                        cast(pd.Series, sub[(sub["layer"] == last_layer) & (sub["module"] == "rc2")]["accuracy"]).iloc[
                            0
                        ]
                    ),
                    "fixed": float(
                        cast(pd.Series, sub[(sub["layer"] == last_layer) & (sub["module"] == "act")]["accuracy"]).iloc[
                            0
                        ]
                    ),
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
    large_df = cast(
        pd.DataFrame, df[df["model_size"] == df["model_name"].map(lambda value: REPRESENTATIVE_VFMs[str(value)])]
    )
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

    path = RUN_RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep vision models and hard datasets
    pattern = "|".join(VFMs)
    df = cast(pd.DataFrame, df[df["model_name"].str.contains(pattern, case=False, na=False)])
    df = cast(pd.DataFrame, df[df["dataset_name"].isin(hard_datasets)])

    def print_statistics(frame: pd.DataFrame, scope: str) -> None:
        rows = []
        for _, sub in frame.groupby(["dataset_name", "model_name", "model_size"]):
            last_layer = sub["layer"].max()
            rows.append(
                {
                    "standard": float(
                        cast(pd.Series, sub[(sub["layer"] == last_layer) & (sub["module"] == "rc2")]["accuracy"]).iloc[
                            0
                        ]
                    ),
                    "fixed": float(
                        cast(pd.Series, sub[(sub["layer"] == last_layer) & (sub["module"] == "act")]["accuracy"]).iloc[
                            0
                        ]
                    ),
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
    large_df = cast(
        pd.DataFrame, df[df["model_size"] == df["model_name"].map(lambda value: REPRESENTATIVE_VFMs[str(value)])]
    )
    print_statistics(large_df, "Large variants")


def table_ablation_pooling_vision() -> None:
    r"""
    Performance comparison between standard, heuristic and oracle probing on large vision models with a CLS token,
    for each pooling over the sequence of tokens: mean over patch tokens (main results), CLS token, and concatenation
    of the CLS token and the mean over patch tokens.
    """
    # Recover results
    main_df = pd.read_csv(RUN_RESULT_DIR / "cls.csv")
    ablation_df = pd.read_csv(ABLATION_RESULT_DIR / "cls.csv")
    poolings = {
        "content": main_df,
        "cls": ablation_df.loc[ablation_df["ablation"] == "cls"],
        "all": ablation_df.loc[ablation_df["ablation"] == "cls_concat"],
    }

    # Results on representative models with a CLS token (AIMv2 has none)
    family_order = ["vit", "dinov2", "dinov3"]
    rows: list[dict[str, object]] = []
    for pooling, df in poolings.items():
        for family in family_order:
            for dataset in REPRESENTATIVE_DATASETS:
                mask = (
                    (df["model_name"] == family)
                    & (df["model_size"] == REPRESENTATIVE_VFMs[family])
                    & (df["dataset_name"] == dataset)
                )
                sub = df.loc[mask]
                last_layer = int(sub["layer"].max())
                standard = float(sub.loc[(sub["layer"] == last_layer) & (sub["module"] == "rc2"), "accuracy"].iloc[0])
                fixed = float(sub.loc[(sub["layer"] == last_layer) & (sub["module"] == "act"), "accuracy"].iloc[0])
                oracle = float(sub["accuracy"].max())
                rows.append(
                    {
                        "pooling": pooling,
                        "family": family,
                        "dataset": dataset,
                        "standard": standard,
                        "fixed": fixed - standard,
                        "oracle": oracle - standard,
                    }
                )
    summary = pd.DataFrame(rows)

    def format_gain(value: float) -> str:
        if round(value, 1) > 0:
            return rf"\add{{+{value:.1f}}}"
        if round(value, 1) < 0:
            return rf"\sub{{{value:.1f}}}"
        return "0.0"

    def format_accuracy(value: float) -> str:
        return f"{value:.1f}"

    # LaTeX NiceTabular: one group of columns per dataset, one column per pooling
    pooling_labels = {"content": "Mean", "cls": "CLS", "all": "Concat"}
    datasets = REPRESENTATIVE_DATASETS
    n_poolings = len(pooling_labels)
    n_columns = 1 + n_poolings * len(datasets)
    lines = [rf"\begin{{NiceTabular}}{{l{'c' * (n_columns - 1)}}}", r"\toprule"]
    groups = [DATASETS[dataset] for dataset in datasets]
    lines.append(
        r"\multirow{2}{*}{Strategy} & "
        + " & ".join(rf"\multicolumn{{{n_poolings}}}{{c}}{{{group}}}" for group in groups)
        + r" \\"
    )
    lines.append(
        " ".join(rf"\cmidrule(l){{{2 + n_poolings * k}-{1 + n_poolings * (k + 1)}}}" for k in range(len(datasets)))
    )
    lines.append(" & " + " & ".join(list(pooling_labels.values()) * len(datasets)) + r" \\")
    lines.append(r"\midrule")
    for family in family_order:
        family_df = summary.loc[summary["family"] == family]
        lines.append(rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{{VFM_NAMES[family]}}}}} \\")
        lines.extend([r"\arrayrulecolor{gray!50}", r"\midrule"])
        for strategy, label in [
            ("standard", r"\texttt{Baseline}"),
            ("fixed", r"$\vfm$"),
            ("oracle", r"\oraclerow \texttt{Oracle}"),
        ]:
            formatter = format_accuracy if strategy == "standard" else format_gain
            values = []
            for dataset in datasets:
                for pooling in pooling_labels:
                    pooling_df = family_df.loc[family_df["pooling"] == pooling]
                    value = float(pooling_df.loc[pooling_df["dataset"] == dataset, strategy].iloc[0])
                    values.append(formatter(value))
            lines.append(f"{label} & " + " & ".join(values) + r" \\")
        lines.append(r"\arrayrulecolor{black}")
        if family != family_order[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{NiceTabular}"])
    print("\n".join(lines))


def table_ablation_registers_vision() -> None:
    r"""
    Performance comparison between standard, heuristic and oracle probing on DINOv2-L without and with register
    tokens [1]_, which are excluded from the pooling as the CLS token.

    References
    ----------
    .. [1] T. Darcet et al. Vision Transformers Need Registers. In ICLR 2024
    """
    # Recover results
    main_df = pd.read_csv(RUN_RESULT_DIR / "cls.csv")
    ablation_df = pd.read_csv(ABLATION_RESULT_DIR / "cls.csv")
    models = {
        "DINOv2-L": main_df.loc[(main_df["model_name"] == "dinov2") & (main_df["model_size"] == "large")],
        "DINOv2-L with registers": ablation_df.loc[ablation_df["ablation"] == "registers"],
    }

    # Results
    rows: list[dict[str, object]] = []
    for model, df in models.items():
        for dataset in REPRESENTATIVE_DATASETS:
            sub = df.loc[df["dataset_name"] == dataset]
            last_layer = int(sub["layer"].max())
            standard = float(sub.loc[(sub["layer"] == last_layer) & (sub["module"] == "rc2"), "accuracy"].iloc[0])
            fixed = float(sub.loc[(sub["layer"] == last_layer) & (sub["module"] == "act"), "accuracy"].iloc[0])
            oracle = float(sub["accuracy"].max())
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "standard": standard,
                    "fixed": fixed - standard,
                    "oracle": oracle - standard,
                }
            )
    summary = pd.DataFrame(rows)

    def format_gain(value: float) -> str:
        if round(value, 1) > 0:
            return rf"\add{{+{value:.1f}}}"
        if round(value, 1) < 0:
            return rf"\sub{{{value:.1f}}}"
        return "0.0"

    def format_accuracy(value: float) -> str:
        return f"{value:.1f}"

    # LaTeX NiceTabular
    header = ["Strategy"] + [DATASETS[dataset] for dataset in REPRESENTATIVE_DATASETS]
    n_columns = len(header)
    lines = [rf"\begin{{NiceTabular}}{{l{'c' * (n_columns - 1)}}}", r"\toprule", " & ".join(header) + r" \\"]
    lines.append(r"\midrule")
    for model in models:
        model_df = summary.loc[summary["model"] == model]
        lines.append(rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{{model}}}}} \\")
        lines.extend([r"\arrayrulecolor{gray!50}", r"\midrule"])
        for strategy, label in [
            ("standard", r"\texttt{Baseline}"),
            ("fixed", r"$\vfm$"),
            ("oracle", r"\oraclerow \texttt{Oracle}"),
        ]:
            formatter = format_accuracy if strategy == "standard" else format_gain
            values = [
                formatter(float(model_df.loc[model_df["dataset"] == dataset, strategy].iloc[0]))
                for dataset in REPRESENTATIVE_DATASETS
            ]
            lines.append(f"{label} & " + " & ".join(values) + r" \\")
        lines.append(r"\arrayrulecolor{black}")
        if model != list(models)[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{NiceTabular}"])
    print("\n".join(lines))


def table_ablation_clip_vision() -> None:
    r"""
    Performance comparison between standard, heuristic and oracle probing on CLIP-L [1]_, a contrastive
    image-text model, used as a vision encoder.

    References
    ----------
    .. [1] A. Radford, J. W. Kim et al. Learning Transferable Visual Models From Natural Language
           Supervision. In ICML 2021
    """
    # Recover results
    ablation_df = pd.read_csv(ABLATION_RESULT_DIR / "cls.csv")
    df = ablation_df.loc[ablation_df["ablation"] == "clip"]

    # Results
    rows: list[dict[str, object]] = []
    for dataset in REPRESENTATIVE_DATASETS:
        sub = df.loc[df["dataset_name"] == dataset]
        last_layer = int(sub["layer"].max())
        standard = float(sub.loc[(sub["layer"] == last_layer) & (sub["module"] == "rc2"), "accuracy"].iloc[0])
        fixed = float(sub.loc[(sub["layer"] == last_layer) & (sub["module"] == "act"), "accuracy"].iloc[0])
        oracle = float(sub["accuracy"].max())
        rows.append({"dataset": dataset, "standard": standard, "fixed": fixed - standard, "oracle": oracle - standard})
    summary = pd.DataFrame(rows)

    def format_gain(value: float) -> str:
        if round(value, 1) > 0:
            return rf"\add{{+{value:.1f}}}"
        if round(value, 1) < 0:
            return rf"\sub{{{value:.1f}}}"
        return "0.0"

    def format_accuracy(value: float) -> str:
        return f"{value:.1f}"

    # LaTeX NiceTabular
    header = ["Strategy"] + [DATASETS[dataset] for dataset in REPRESENTATIVE_DATASETS]
    n_columns = len(header)
    lines = [rf"\begin{{NiceTabular}}{{l{'c' * (n_columns - 1)}}}", r"\toprule", " & ".join(header) + r" \\"]
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{CLIP-L}}}} \\")
    lines.extend([r"\arrayrulecolor{gray!50}", r"\midrule"])
    for strategy, label in [
        ("standard", r"\texttt{Baseline}"),
        ("fixed", r"$\vfm$"),
        ("oracle", r"\oraclerow \texttt{Oracle}"),
    ]:
        formatter = format_accuracy if strategy == "standard" else format_gain
        values = [
            formatter(float(summary.loc[summary["dataset"] == dataset, strategy].iloc[0]))
            for dataset in REPRESENTATIVE_DATASETS
        ]
        lines.append(f"{label} & " + " & ".join(values) + r" \\")
    lines.extend([r"\arrayrulecolor{black}", r"\bottomrule", r"\end{NiceTabular}"])
    print("\n".join(lines))


def table_ablation_final_norm_vision() -> None:
    r"""
    Performance comparison between standard probing (rc2 at the last layer), standard probing after the final norm
    (rc2norm, i.e. the output of the model) and heuristic probing (act at the last layer) on large vision models.
    """
    # Recover results
    main_df = pd.read_csv(RUN_RESULT_DIR / "cls.csv")
    ablation_df = pd.read_csv(ABLATION_RESULT_DIR / "cls.csv")
    final_norm_df = ablation_df.loc[(ablation_df["ablation"] == "final_norm") & (ablation_df["module"] == "rc2norm")]

    # Results on representative models
    family_order = list(REPRESENTATIVE_VFMs.keys())
    rows: list[dict[str, object]] = []
    for family in family_order:
        for dataset in REPRESENTATIVE_DATASETS:
            mask = (
                (main_df["model_name"] == family)
                & (main_df["model_size"] == REPRESENTATIVE_VFMs[family])
                & (main_df["dataset_name"] == dataset)
            )
            sub = main_df.loc[mask]
            last_layer = int(sub["layer"].max())
            standard = float(sub.loc[(sub["layer"] == last_layer) & (sub["module"] == "rc2"), "accuracy"].iloc[0])
            fixed = float(sub.loc[(sub["layer"] == last_layer) & (sub["module"] == "act"), "accuracy"].iloc[0])
            mask = (
                (final_norm_df["model_name"] == family)
                & (final_norm_df["model_size"] == REPRESENTATIVE_VFMs[family])
                & (final_norm_df["dataset_name"] == dataset)
            )
            final_norm = float(final_norm_df.loc[mask, "accuracy"].iloc[0])
            rows.append(
                {
                    "family": family,
                    "dataset": dataset,
                    "standard": standard,
                    "final_norm": final_norm - standard,
                    "fixed": fixed - standard,
                }
            )
    summary = pd.DataFrame(rows)

    def format_gain(value: float) -> str:
        if round(value, 1) > 0:
            return rf"\add{{+{value:.1f}}}"
        if round(value, 1) < 0:
            return rf"\sub{{{value:.1f}}}"
        return "0.0"

    def format_accuracy(value: float) -> str:
        return f"{value:.1f}"

    # LaTeX NiceTabular
    header = ["Strategy"] + [DATASETS[dataset] for dataset in REPRESENTATIVE_DATASETS]
    n_columns = len(header)
    lines = [rf"\begin{{NiceTabular}}{{l{'c' * (n_columns - 1)}}}", r"\toprule", " & ".join(header) + r" \\"]
    lines.append(r"\midrule")
    for family in family_order:
        family_df = summary.loc[summary["family"] == family]
        lines.append(rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{{VFM_NAMES[family]}}}}} \\")
        lines.extend([r"\arrayrulecolor{gray!50}", r"\midrule"])
        for strategy, label in [
            ("standard", r"\texttt{Baseline}"),
            ("final_norm", r"\texttt{RC2Norm@100}"),
            ("fixed", r"$\vfm$"),
        ]:
            formatter = format_accuracy if strategy == "standard" else format_gain
            values = [
                formatter(float(family_df.loc[family_df["dataset"] == dataset, strategy].iloc[0]))
                for dataset in REPRESENTATIVE_DATASETS
            ]
            lines.append(f"{label} & " + " & ".join(values) + r" \\")
        lines.append(r"\arrayrulecolor{black}")
        if family != family_order[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{NiceTabular}"])
    print("\n".join(lines))


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

    # Recover aggregate results.
    path = RUN_RESULT_DIR / "summary"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep language models
    df = cast(pd.DataFrame, df[df["model_name"].isin(LLMs)])
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    gain_types = ["gain", "gain_module", "gain_layer"]
    gain_labels = ["oracle", "module", "layer"]
    model_labels = {
        "qwen3": ["Qwen3 1.7B", "Qwen3 4B", "Qwen3 8B"],
        "llama3": ["Llama3 1B", "Llama3 3B", "Llama3 8B"],
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
        gain_map = dict(zip(gain_types, gain_labels, strict=False))
        agg_long["gain_type"] = agg_long["gain_type"].map(gain_map)
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
            ax.set_ylabel("Gain over probing baseline (pp)", fontsize=FONTSIZE)
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
    path = RUN_RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep language models
    df = cast(pd.DataFrame, df[df["model_name"].isin(LLMs)])
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    model_labels = {"qwen3": "Qwen3 1.7B", "llama3": "Llama3 1B", "gemma3": "Gemma3 1B"}
    tasks = df["task_name"].unique()
    for i, (model_family, model_name) in enumerate(REPRESENTATIVE_LLMs.items()):
        ax = axes[i]
        for module in ["rc2", "ln2"]:
            subset = df[(df["model_name"] == model_name) & (df["module"] == module)].copy()
            depth = []
            values = []
            for task_name in tasks:
                df_task = subset[subset.task_name == task_name]
                df_task = cast(pd.DataFrame, df_task).sort_values(by="layer")
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
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 1)
        tol = 0
        ax.set_yticks(np.linspace(ymin - tol, ymax - tol, 3).astype(int))
        ax.set_xlabel(r"Layer depth ($\%$)", fontsize=FONTSIZE)
        if i == 0:
            ax.set_ylabel(r"Accuracy ($\%$)", fontsize=FONTSIZE)
        ax.set_title(f"{model_labels[model_family]}", fontsize=FONTSIZE)
    sns.despine(fig, axes[0], trim=True, right=True, offset=10)

    # Common legend
    legend = axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.63, 0.58),
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
    height = 4.4
    nrows = 3
    ncols = 3
    figsize = ncols * width, nrows * height
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)

    # Recover results
    path = RUN_RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep language models
    df = cast(pd.DataFrame, df[df["model_name"].isin(LLMs)])
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    model_labels = {
        "qwen3": ["Qwen3 1.7B", "Qwen3 4B", "Qwen3 8B"],
        "llama3": ["Llama3 1B", "Llama3 3B", "Llama3 8B"],
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
                    df_task = cast(pd.DataFrame, df_task).sort_values(by="layer")
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
            ymin, ymax = ax.get_ylim()
            tol = 1
            ax.set_yticks(np.linspace(ymin - tol, ymax - tol, 3).astype(int))
            if i == nrows - 1:
                ax.set_xlabel(r"Layer depth ($\%$)", fontsize=FONTSIZE)
            if j == 0:
                ax.set_ylabel(r"Accuracy ($\%$)", fontsize=FONTSIZE)
            ax.set_title(model_labels[model_family][j], fontsize=FONTSIZE)
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
        figname = f"probing_depth_all_language_{filename}"
        save_plot(figname=figname)


def plot_oracle_recovery_language(filename: str, save: bool = False) -> None:
    r"""
    Recover the table of oracle gain recovery (computed as gain / oracle_gain) averaged over models.

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
    path = RUN_RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()
    df = df.groupby(["model_name", "layer", "module"])["accuracy"].mean().reset_index()

    # Keep LLM models
    all_models = [m for models in LLM_FAMILIES.values() for m in models]
    df = cast(pd.DataFrame, df[df["model_name"].isin(all_models)])
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
        seed=0,
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
    ax.set_xticklabels([r"Best $\mathtt{RC2}$", r"$\mathtt{LN2@60}$", r"Best $\mathtt{LN2}$"])
    ax.set_ylim(20, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xlabel("")
    ax.set_ylabel(r"Oracle gain recovered ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.tight_layout()
    if save:
        save_plot(figname=f"oracle_recovery_language_{filename}")


def plot_probing_depth_post_training_language(save: bool = False) -> None:
    r"""
    Evolution of the ln2 and rc2 probing accuracy along with the depth for Qwen3 before (pretrained, Base models,
    top row) and after post-training (bottom row), with the accuracy averaged over tasks.

    Parameters
    ----------
    save: bool, default=False
        Whether to save the plot.
    """
    # Figure
    width = 5.5
    height = 5
    sizes = ["1.7B", "4B", "8B"]
    nrows = 2
    ncols = len(sizes)
    figsize = ncols * width, nrows * height
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey="col")

    # Recover results
    main_df = pd.read_csv(RUN_RESULT_DIR / "mteb.csv")
    ablation_df = pd.read_csv(ABLATION_RESULT_DIR / "mteb.csv")
    stages = {
        "Pretrained": (ablation_df.loc[ablation_df["ablation"] == "qwen3_base"], "-Base"),
        "Post-trained": (main_df, ""),
    }

    # Results
    for i, (stage, (df, suffix)) in enumerate(stages.items()):
        for j, size in enumerate(sizes):
            ax = axes[i, j]
            model_name = f"Qwen3-{size}{suffix}"
            for module in ["rc2", "ln2"]:
                subset = df.loc[(df["model_name"] == model_name) & (df["module"] == module)]
                acc = subset.groupby("layer")["accuracy"].mean().sort_index()
                depth = 100 * acc.index.to_numpy() / acc.index.max()
                ax.plot(
                    depth,
                    acc.to_numpy(),
                    color=MODULE_PALETTE[module],
                    linewidth=LINEWIDTH,
                    linestyle="-" if module != "rc2" else "--",
                    label=module.upper(),
                )

                # Mark the highest ln2 performance
                if module == "ln2":
                    index = int(np.argmax(acc.to_numpy()))
                    ax.scatter(
                        depth[index],
                        acc.to_numpy()[index],
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
            ax.tick_params(
                axis="both",
                direction="out",
                length=5,
                width=1,
                labelsize=FONTSIZE,
                labelbottom=i == nrows - 1,
            )
            if i == 0:
                ax.set_title(f"Qwen3 {size}\n", fontsize=FONTSIZE)
            if i == nrows - 1:
                ax.set_xlabel(r"Layer depth ($\%$)", fontsize=FONTSIZE)
            if j == 0:
                ax.set_ylabel(f"{stage}\n" + r"Accuracy ($\%$)", fontsize=FONTSIZE)
    sns.despine(fig, axes[0, 0], trim=True, right=True, offset=10)

    # Common legend
    legend = axes[0, 0].legend(
        loc="lower right",
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
        figname = "probing_depth_post_training_language"
        save_plot(figname=figname, dirname="ablations")


# ------------------------------------------------------------------------------
# LLMs table functions
# ------------------------------------------------------------------------------
def table_performance_comparison_language(filename: str) -> None:
    r"""Performance comparison between standard probing, heuristic probing and oracle probing."""
    # Recover results
    path = RUN_RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()
    df = df.groupby(["model_name", "layer", "module"])["accuracy"].mean().reset_index()

    # Keep LLM models
    all_models = [m for models in LLM_FAMILIES.values() for m in models]
    df = cast(pd.DataFrame, df[df["model_name"].isin(all_models)])
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    rows = []
    groups = "model_name"
    for model_name, sub in df.groupby(groups):
        last_layer = sub.layer.max()
        standard = sub[(sub.layer == last_layer) & (sub.module == "rc2")].accuracy.iloc[0]
        heuristic_depth = 0.6
        heuristic_layer = int(heuristic_depth * last_layer)
        standard_heuristic = sub[(sub.layer == heuristic_layer) & (sub.module == "rc2")].accuracy.iloc[0]
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
                    "strategy": "Standard60",
                    "accuracy": standard_heuristic,
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
    table = table.loc[["Standard", "Standard60", "Fixed", "Oracle"]]

    def format_accuracy(value: float) -> str:
        return f"{value:.1f}"

    def format_gain(value: float) -> str:
        if round(value, 1) > 0:
            return rf"\addsmall{{+{value:.1f}}}"
        if round(value, 1) < 0:
            return rf"\subsmall{{{value:.1f}}}"
        return "0.0"

    # Model order and displayed sizes
    model_order = [model for models in LLM_FAMILIES.values() for model in models]
    model_sizes = ["1.7B", "4B", "8B", "1B", "3B", "8B", "1B", "4B", "12B"]

    # LaTeX NiceTabular
    lines = []
    lines.append(
        r"\begin{NiceTabular}{"
        r"@{} l "
        r"*{9}{w{r}{2.8em}"
        r"@{\hspace{3pt}}"
        r"w{r}{1.7em}"
        r"@{\hspace{1pt}}"
        r"}@{}"
        r"}"
    )
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{Strategy} & "
        r"\multicolumn{6}{c}{\textit{Qwen3}} & "
        r"\multicolumn{6}{c}{\textit{Llama3}} & "
        r"\multicolumn{6}{c}{\textit{Gemma3}} \\"
    )
    lines.append(r"\cmidrule(l){2-7} \cmidrule(l){8-13} \cmidrule(l){14-19}")
    lines.append(" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{size}}}" for size in model_sizes) + r" \\")
    lines.append(r"\midrule")

    # Baseline
    values = []
    for model in model_order:
        standard = float(cast(Any, table.loc["Standard", model]))
        values.extend([format_accuracy(standard), ""])
    lines.append(r"\texttt{Baseline} & " + " & ".join(values) + r" \\")

    # Baseline heuristic
    values = []
    for model in model_order:
        standard = float(cast(Any, table.loc["Standard", model]))
        fixed = float(cast(Any, table.loc["Standard60", model]))
        values.extend(
            [
                format_accuracy(fixed),
                format_gain(fixed - standard),
            ]
        )
    lines.append(r"\texttt{RC2@60} & " + " & ".join(values) + r" \\")

    # Fixed strategy
    values = []
    for model in model_order:
        standard = float(cast(Any, table.loc["Standard", model]))
        fixed = float(cast(Any, table.loc["Fixed", model]))
        values.extend(
            [
                format_accuracy(fixed),
                format_gain(fixed - standard),
            ]
        )
    lines.append(r"$\llm$ & " + " & ".join(values) + r" \\")

    # Oracle
    values = []
    for model in model_order:
        standard = float(cast(Any, table.loc["Standard", model]))
        oracle = float(cast(Any, table.loc["Oracle", model]))
        values.extend(
            [
                format_accuracy(oracle),
                format_gain(oracle - standard),
            ]
        )
    lines.append(r"\oraclerow \texttt{Oracle} & " + " & ".join(values) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{NiceTabular}")

    print("\n".join(lines))


def table_performance_comparison_task_language(filename: str) -> None:
    r"""
    Generate the LaTeX NiceTabular environment for LLM probing results, per task,
    averaged over models.

    For each model:
    - Baseline: RC2 at the last layer.
    - RC2@60: RC2 at 60% depth.
    - LN2@60: LN2 at 60% depth.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    """
    # Recover results
    df = pd.read_csv(RUN_RESULT_DIR / f"{filename}.csv").copy()

    # Keep LLM models
    df = df[df["model_name"].isin(LLMs)]

    # Results
    rows = []
    for (task, model), sub in df.groupby(["task_name", "model_name"]):
        last_layer = sub.layer.max()
        heuristic_layer = int(0.6 * last_layer)

        def acc(layer: int, module: str, sub: pd.DataFrame = sub) -> float:
            return sub[(sub.layer == layer) & (sub.module == module)].accuracy.iloc[0]

        rows.append(
            {
                "task": task,
                "model": model,
                "standard": acc(last_layer, "rc2"),
                "standard_heuristic": acc(heuristic_layer, "rc2"),
                "fixed": acc(heuristic_layer, "ln2"),
            }
        )

    summary = pd.DataFrame(rows)

    # Average over models for each task
    summary = summary.groupby("task")[["standard", "standard_heuristic", "fixed"]].mean().reset_index()
    ordered_tasks = sorted(summary["task"].tolist(), key=str.lower)

    # Format values
    def format_accuracy(value: float) -> str:
        return f"{value:.1f}"

    def format_gain(value: float) -> str:
        if round(value, 1) > 0:
            return rf"\addsmall{{+{value:.1f}}}"
        if round(value, 1) < 0:
            return rf"\subsmall{{{value:.1f}}}"
        return "0.0"

    # Header
    header = [
        "Task",
        r"\multicolumn{2}{c}{\texttt{Baseline}}",
        r"\multicolumn{2}{c}{\texttt{RC2@60}}",
        r"\multicolumn{2}{c}{$\llm$}",
    ]

    # LaTeX table
    lines = [
        r"\begin{NiceTabular}{"
        r"@{} l "
        r"*{3}{w{r}{2.8em}@{\hspace{6pt}}w{r}{2.2em}@{\hspace{4pt}}}"
        r"@{}"
        r"}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]

    # Task rows
    for task in ordered_tasks:
        row = summary[summary["task"] == task].iloc[0]
        standard = float(row["standard"])
        standard_heuristic = float(row["standard_heuristic"])
        fixed = float(row["fixed"])
        lines.append(
            rf"{task} & "
            rf"{format_accuracy(standard)} & & "
            rf"{format_accuracy(standard_heuristic)} & {format_gain(standard_heuristic - standard)} & "
            rf"{format_accuracy(fixed)} & {format_gain(fixed - standard)} \\"
        )

    lines += [r"\bottomrule", r"\end{NiceTabular}"]
    print("\n".join(lines))


def table_performance_comparison_all_language(filename: str, model_family: str) -> None:
    r"""
    Generate the LaTeX NiceTabular environment for task-level LLM results.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    model_family: str
        LLM family to report, e.g. "Qwen3", "Llama3", or "Gemma3".
    """
    # Recover results
    df = pd.read_csv(RUN_RESULT_DIR / f"{filename}.csv").copy()

    # Model configuration: checkpoint name -> displayed name (consistent with the main text)
    model_names = {
        "Qwen3": {"Qwen3-1.7B": "Qwen3 1.7B", "Qwen3-4B": "Qwen3 4B", "Qwen3-8B": "Qwen3 8B"},
        "Llama3": {"Llama-3.2-1B": "Llama3 1B", "Llama-3.2-3B": "Llama3 3B", "Llama-3.1-8B": "Llama3 8B"},
        "Gemma3": {"gemma-3-1b-pt": "Gemma3 1B", "gemma-3-4b-pt": "Gemma3 4B", "gemma-3-12b-pt": "Gemma3 12B"},
    }
    if model_family not in model_names:
        raise ValueError(f"Unknown model family '{model_family}'. Expected one of {list(model_names)}.")
    models = list(model_names[model_family])

    # Keep models from the selected family
    df = df[df["model_name"].isin(models)].copy()

    # Results
    rows = []
    for (task, model), sub in df.groupby(["task_name", "model_name"]):
        last_layer = sub.layer.max()
        heuristic_layer = int(0.6 * last_layer)

        def acc(layer: int, module: str, sub: pd.DataFrame = sub) -> float:
            return sub[(sub.layer == layer) & (sub.module == module)].accuracy.iloc[0]

        rows.append(
            {
                "task": task,
                "model": model,
                "standard": acc(last_layer, "rc2"),
                "standard_heuristic": acc(heuristic_layer, "rc2"),
                "fixed": acc(heuristic_layer, "ln2"),
            }
        )
    summary = pd.DataFrame(rows)

    # Task order (case-insensitive, consistent across tables)
    task_order = sorted(summary["task"].unique(), key=str.lower)

    def format_accuracy(value: float) -> str:
        return f"{value:.1f}"

    # LaTeX table
    lines = [
        r"\begin{NiceTabular}{"
        r"@{} l "
        r"*{3}{w{c}{3.6em}@{\hspace{12pt}}}"
        r"*{3}{w{c}{3.6em}@{\hspace{12pt}}}"
        r"*{3}{w{c}{3.6em}@{\hspace{12pt}}}"
        r"@{}"
        r"}",
        r"\toprule",
    ]

    # Model header
    model_header = [r"\multirow{2}{*}{Task}"]
    for model in models:
        model_header.append(rf"\multicolumn{{3}}{{c}}{{\textit{{{model_names[model_family][model]}}}}}")
    lines.append(" & ".join(model_header) + r" \\")
    lines.append(r"\cmidrule(l){2-4} \cmidrule(l){5-7} \cmidrule(l){8-10}")

    # Strategy header
    strategy_header = [""]
    for _ in models:
        strategy_header.extend([r"\texttt{Baseline}", r"\texttt{RC2@60}", r"$\llm$"])
    lines.append(" & ".join(strategy_header) + r" \\")
    lines.append(r"\midrule")

    # Task rows
    for task in task_order:
        row_values = [task]
        for model in models:
            sub = summary[(summary["task"] == task) & (summary["model"] == model)]
            if sub.empty:
                row_values.extend(["", "", ""])
                continue
            result = sub.iloc[0]
            row_values.extend(
                [
                    format_accuracy(float(result["standard"])),
                    format_accuracy(float(result["standard_heuristic"])),
                    format_accuracy(float(result["fixed"])),
                ]
            )
        lines.append(" & ".join(row_values) + r" \\")

    lines += [r"\bottomrule", r"\end{NiceTabular}"]
    print("\n".join(lines))


def table_global_statistics_language(filename: str) -> None:
    r"""
    Compute global statistics for fixed, depth-optimized, and oracle language strategies.

    All searched locations (best RC2 layer, best LN2 layer, oracle layer-module pair) are
    selected per model on the MTEB average over tasks (benchmark level), then evaluated on
    each task. This matches the oracle definition of Table 2 and Figure 9.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    """
    # Recover results
    df = pd.read_csv(RUN_RESULT_DIR / f"{filename}.csv").copy()

    # Keep LLM models
    all_models = [m for models in LLM_FAMILIES.values() for m in models]
    df = cast(pd.DataFrame, df[df["model_name"].isin(all_models)])
    logger.info(f"Results are presented over a total of {len(df)} probing runs.")

    # Sanity check: each (model, layer, module) must be evaluated on all tasks
    n_tasks = df["task_name"].nunique()
    counts = df.groupby(["model_name", "layer", "module"])["task_name"].nunique()
    if (counts != n_tasks).any():
        raise ValueError(f"Incomplete evaluations:\n{counts[counts != n_tasks]}")

    # Benchmark-level accuracy (average over tasks) for each model and location
    bench = df.groupby(["model_name", "layer", "module"], as_index=False)["accuracy"].mean()

    def best_location(model_name: str, module: str | None = None) -> tuple[int, str]:
        r"""Location maximizing the benchmark accuracy, optionally restricted to a module."""
        sub = bench[bench["model_name"] == model_name]
        if module is not None:
            sub = sub[sub["module"] == module]
        best = sub.loc[sub["accuracy"].idxmax()]
        return int(best["layer"]), str(best["module"])

    heuristic_depth = 0.6
    last_layers = df.groupby("model_name")["layer"].max()
    locations = {}
    for model_name in df["model_name"].unique():
        last_layer = int(last_layers[model_name])
        locations[model_name] = {
            "standard": (last_layer, "rc2"),
            "fixed": (int(heuristic_depth * last_layer), "ln2"),
            "best_rc2": best_location(model_name, "rc2"),
            "best_ln2": best_location(model_name, "ln2"),
            "oracle": best_location(model_name),
        }
        logger.info(f"{model_name}: oracle location = {locations[model_name]['oracle']}")

    strategies = ["standard", "fixed", "best_rc2", "best_ln2", "oracle"]

    # Per model-task evaluations at the fixed (per-model) locations
    rows = []
    for key_values, sub in cast(Any, df.groupby(["task_name", "model_name"])):
        task_name, model_name = key_values
        row = {"task": task_name, "model_name": model_name}
        for strategy in strategies:
            layer, module = locations[model_name][strategy]
            row[strategy] = float(
                cast(pd.Series, sub[(sub["layer"] == layer) & (sub["module"] == module)]["accuracy"]).iloc[0]
            )
        rows.append(row)

    evaluations = pd.DataFrame(rows)
    tasks = evaluations.groupby("task")[strategies].mean()
    models = evaluations.groupby("model_name")[strategies].mean()  # benchmark level, = Table 2

    def print_statistics(strategy: str, label: str) -> None:
        evaluation_wins = (evaluations[strategy] > evaluations["standard"]).sum()
        evaluation_ties = (evaluations[strategy] == evaluations["standard"]).sum()
        task_wins = (tasks[strategy] > tasks["standard"]).sum()
        model_wins = (models[strategy] > models["standard"]).sum()
        evaluation_gain = (evaluations[strategy] - evaluations["standard"]).mean()
        task_gain = (tasks[strategy] - tasks["standard"]).mean()
        model_gain = (models[strategy] - models["standard"]).mean()
        print(
            f"{label} model-task evaluations where strategy improves: "
            f"{evaluation_wins}/{len(evaluations)} ({100 * evaluation_wins / len(evaluations):.1f}%), "
            f"ties: {evaluation_ties}"
        )
        print(
            f"{label} tasks where strategy improves on average: "
            f"{task_wins}/{len(tasks)} ({100 * task_wins / len(tasks):.1f}%)"
        )
        print(f"{label} models where strategy improves: {model_wins}/{len(models)}")
        print(f"{label} average gain over model-task evaluations: {evaluation_gain:.2f} pp")
        print(f"{label} average gain over tasks: {task_gain:.2f} pp")
        print(f"{label} average gain over models (benchmark level): {model_gain:.2f} pp")

    print("\nFixed module comparison: LN2@60 vs. standard RC2@100.")
    print_statistics("fixed", "Fixed LN2")
    print("\nDepth-optimized module comparison (layer selected on the benchmark average).")
    print_statistics("best_rc2", "Depth-RC2")
    print_statistics("best_ln2", "Depth-LN2")
    print("\nOracle comparison (layer-module pair selected on the benchmark average).")
    print_statistics("oracle", "Oracle")

    # Oracle gain recovery at the benchmark level (Figure 9)
    oracle_gain = models["oracle"] - models["standard"]
    print("\nRecovery of the benchmark-level oracle gain, averaged over models:")
    for strategy, label in [("best_rc2", "Best RC2"), ("fixed", "LN2@60"), ("best_ln2", "Best LN2")]:
        recovery = (models[strategy] - models["standard"]) / oracle_gain
        print(
            f"{label}: {100 * recovery.mean():.1f}% (min {100 * recovery.min():.1f}%, max {100 * recovery.max():.1f}%)"
        )


def table_global_statistics_language_(filename: str) -> None:
    r"""
    Compute global statistics for fixed and oracle language strategies.

    Parameters
    ----------
    filename: str
        Name of the result file to use.
    """
    # Recover results
    path = RUN_RESULT_DIR
    df = pd.read_csv(path / f"{filename}.csv").copy()

    # Keep LLM models
    all_models = [m for models in LLM_FAMILIES.values() for m in models]
    df = cast(pd.DataFrame, df[df["model_name"].isin(all_models)])
    logger.info(f"Results are presented over a total of {len(df)} runs.")

    # Results
    heuristic_depth = 0.6
    last_layers = df.groupby("model_name")["layer"].max()
    rows = []
    for key_values, sub in cast(Any, df.groupby(["task_name", "model_name"])):
        task_name, model_name = key_values
        last_layer = last_layers[model_name]
        standard = float(
            cast(pd.Series, sub[(sub["layer"] == last_layer) & (sub["module"] == "rc2")]["accuracy"]).iloc[0]
        )
        heuristic_layer = int(heuristic_depth * last_layer)
        fixed = float(
            cast(pd.Series, sub[(sub["layer"] == heuristic_layer) & (sub["module"] == "ln2")]["accuracy"]).iloc[0]
        )
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


def table_ablation_final_norm_language() -> None:
    r"""
    Performance comparison between standard probing (rc2 at the last layer), standard probing after the final norm
    (rc2norm, i.e. the output of the model) and heuristic probing (ln2 at 60% of the depth) on LLMs, with the
    accuracy averaged over tasks.
    """
    # Recover results
    main_df = pd.read_csv(RUN_RESULT_DIR / "mteb.csv")
    main_df = main_df.groupby(["model_name", "layer", "module"])["accuracy"].mean().reset_index()
    ablation_df = pd.read_csv(ABLATION_RESULT_DIR / "mteb.csv")
    ablation_df = ablation_df.loc[(ablation_df["ablation"] == "final_norm") & (ablation_df["module"] == "rc2norm")]
    final_norm = ablation_df.groupby("model_name")["accuracy"].mean()

    # Results
    model_order = [model for models in LLM_FAMILIES.values() for model in models]
    summary: dict[str, dict[str, float]] = {}
    for model_name in model_order:
        sub = main_df.loc[main_df["model_name"] == model_name]
        last_layer = int(sub["layer"].max())
        heuristic_layer = int(0.6 * last_layer)
        standard = float(sub.loc[(sub["layer"] == last_layer) & (sub["module"] == "rc2"), "accuracy"].iloc[0])
        fixed = float(sub.loc[(sub["layer"] == heuristic_layer) & (sub["module"] == "ln2"), "accuracy"].iloc[0])
        summary[model_name] = {
            "standard": standard,
            "final_norm": float(final_norm[model_name]),
            "fixed": fixed,
        }

    def format_accuracy(value: float) -> str:
        return f"{value:.1f}"

    def format_gain(value: float) -> str:
        if round(value, 1) > 0:
            return rf"\addsmall{{+{value:.1f}}}"
        if round(value, 1) < 0:
            return rf"\subsmall{{{value:.1f}}}"
        return "0.0"

    # Displayed sizes
    model_sizes = ["1.7B", "4B", "8B", "1B", "3B", "8B", "1B", "4B", "12B"]

    # LaTeX NiceTabular
    lines = []
    lines.append(
        r"\begin{NiceTabular}{"
        r"@{} l "
        r"*{9}{w{r}{2.8em}"
        r"@{\hspace{3pt}}"
        r"w{r}{1.7em}"
        r"@{\hspace{1pt}}"
        r"}@{}"
        r"}"
    )
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{Strategy} & "
        r"\multicolumn{6}{c}{\textit{Qwen3}} & "
        r"\multicolumn{6}{c}{\textit{Llama3}} & "
        r"\multicolumn{6}{c}{\textit{Gemma3}} \\"
    )
    lines.append(r"\cmidrule(l){2-7} \cmidrule(l){8-13} \cmidrule(l){14-19}")
    lines.append(" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{size}}}" for size in model_sizes) + r" \\")
    lines.append(r"\midrule")

    # Baseline
    values = []
    for model in model_order:
        values.extend([format_accuracy(summary[model]["standard"]), ""])
    lines.append(r"\texttt{Baseline} & " + " & ".join(values) + r" \\")

    # Final norm and fixed strategy
    for strategy, label in [("final_norm", r"\texttt{RC2Norm@100}"), ("fixed", r"$\llm$")]:
        values = []
        for model in model_order:
            standard = summary[model]["standard"]
            value = summary[model][strategy]
            values.extend([format_accuracy(value), format_gain(value - standard)])
        lines.append(f"{label} & " + " & ".join(values) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{NiceTabular}")
    print("\n".join(lines))


def table_ablation_post_training_language() -> None:
    r"""
    Performance comparison between standard probing (rc2 at the last layer), heuristic probing (ln2 at 60% of the
    depth) and oracle probing on Qwen3 before (pretrained, Base models) and after post-training, with the accuracy
    averaged over tasks.
    """
    # Recover results
    main_df = pd.read_csv(RUN_RESULT_DIR / "mteb.csv")
    ablation_df = pd.read_csv(ABLATION_RESULT_DIR / "mteb.csv")
    df = pd.concat([main_df, ablation_df.loc[ablation_df["ablation"] == "qwen3_base"]])
    df = df.groupby(["model_name", "layer", "module"])["accuracy"].mean().reset_index()

    # Results
    sizes = ["1.7B", "4B", "8B"]
    stages = {"Pretrained": "-Base", "Post-trained": ""}
    model_order = [f"Qwen3-{size}{suffix}" for size in sizes for suffix in stages.values()]
    summary: dict[str, dict[str, float]] = {}
    for model_name in model_order:
        sub = df.loc[df["model_name"] == model_name]
        last_layer = int(sub["layer"].max())
        heuristic_layer = int(0.6 * last_layer)
        standard = float(sub.loc[(sub["layer"] == last_layer) & (sub["module"] == "rc2"), "accuracy"].iloc[0])
        fixed = float(sub.loc[(sub["layer"] == heuristic_layer) & (sub["module"] == "ln2"), "accuracy"].iloc[0])
        oracle = float(sub["accuracy"].max())
        summary[model_name] = {"standard": standard, "fixed": fixed, "oracle": oracle}

    def format_accuracy(value: float) -> str:
        return f"{value:.1f}"

    def format_gain(value: float) -> str:
        if round(value, 1) > 0:
            return rf"\addsmall{{+{value:.1f}}}"
        if round(value, 1) < 0:
            return rf"\subsmall{{{value:.1f}}}"
        return "0.0"

    # LaTeX NiceTabular
    n_models = len(model_order)
    lines = []
    lines.append(
        r"\begin{NiceTabular}{"
        r"@{} l "
        rf"*{{{n_models}}}{{w{{r}}{{2.8em}}"
        r"@{\hspace{3pt}}"
        r"w{r}{1.7em}"
        r"@{\hspace{1pt}}"
        r"}@{}"
        r"}"
    )
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{Strategy} & "
        + " & ".join(rf"\multicolumn{{4}}{{c}}{{\textit{{Qwen3 {size}}}}}" for size in sizes)
        + r" \\"
    )
    lines.append(" ".join(rf"\cmidrule(l){{{2 + 4 * k}-{5 + 4 * k}}}" for k in range(len(sizes))))
    lines.append(" & " + " & ".join([rf"\multicolumn{{2}}{{c}}{{{stage}}}" for stage in stages] * len(sizes)) + r" \\")
    lines.append(r"\midrule")

    # Baseline
    values = []
    for model in model_order:
        values.extend([format_accuracy(summary[model]["standard"]), ""])
    lines.append(r"\texttt{Baseline} & " + " & ".join(values) + r" \\")

    # Fixed strategy and oracle
    for strategy, label in [("fixed", r"$\llm$"), ("oracle", r"\oraclerow \texttt{Oracle}")]:
        values = []
        for model in model_order:
            standard = summary[model]["standard"]
            value = summary[model][strategy]
            values.extend([format_accuracy(value), format_gain(value - standard)])
        lines.append(f"{label} & " + " & ".join(values) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{NiceTabular}")
    print("\n".join(lines))


# ------------------------------------------------------------------------------
# Results functions
# ------------------------------------------------------------------------------
def plot_figures() -> None:
    r"""Intro plots."""
    plot_intro(save=True)

    r"""VFMs plots."""
    # CDF
    save = True
    plot_cdf_vision(filename="cls", save=save)

    # Dataset difficulty
    save = True
    filename = "cls"
    plot_transfer_difficulty_vision(filename=filename, save=save)

    # Cifar10-C
    save = True
    filename = "cifar10_c"
    plot_cifar10c_severity(filename=filename, save=save)
    plot_cifar10c_optimal_depth(filename=filename, save=save)

    # Pretraining scale
    save = True
    filename = "cls"
    plot_pretraining_scale_vision(filename=filename, save=save)

    # Model family
    save = True
    filename = "cls"
    plot_model_family_vision(filename=filename, save=save)

    # Probing depth
    save = True
    filename = "cls"
    plot_probing_depth_vision(filename=filename, save=save)
    for model_name in ["vit", "dinov2", "dinov3", "aimv2"]:
        plot_probing_depth_family_vision(filename=filename, model_name=model_name, save=save)

    # Oracle recovery
    save = True
    filename = "cls"
    plot_oracle_recovery_vision(filename=filename, save=save)

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

    # Oracle recovery
    save = True
    filename = "mteb"
    plot_oracle_recovery_language(filename=filename, save=save)


def get_tables() -> None:
    r"""Number of runs"""
    table_runs()

    r"""VFMs tables."""
    # Winning module frequency
    filename = "cls"
    table_winning_module_frequency_vision(filename=filename)

    # Model size
    filename = "cls"
    table_model_size_vision(filename=filename)

    # Performance comparison
    filename = "cls"
    table_performance_comparison_vision(filename=filename)
    table_performance_comparison_all_vision(filename=filename)
    table_performance_comparison_act_fc1_vision(filename=filename)

    # Global statistics
    filename = "cls"
    table_global_statistics_vision(filename=filename)
    table_global_statistics_ood_vision(filename=filename)

    r"""LLMs tables."""
    # Performance comparison
    filename = "mteb"
    table_performance_comparison_language(filename=filename)
    table_performance_comparison_task_language(filename=filename)
    for model_family in ["Qwen3", "Llama3", "Gemma3"]:
        table_performance_comparison_all_language(filename=filename, model_family=model_family)

    # Global statistics
    filename = "mteb"
    table_global_statistics_language(filename=filename)


def plot_ablation_figures() -> None:
    r"""Ablations plots."""
    save = True
    plot_probing_depth_clip_vision(save=save)
    plot_probing_depth_post_training_language(save=save)


def get_ablation_tables() -> None:
    r"""VFMs ablations tables."""
    table_ablation_pooling_vision()
    table_ablation_registers_vision()
    table_ablation_clip_vision()
    table_ablation_final_norm_vision()

    r"""LLMs ablations tables."""
    table_ablation_final_norm_language()
    table_ablation_post_training_language()


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
            "ablation_plot": plot_ablation_figures,
            "ablation_table": get_ablation_tables,
        }
    )


# %% CLI
if __name__ == "__main__":
    main()
# %%
