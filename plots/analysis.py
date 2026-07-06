r"""
Analyze evaluation results from VFMs, VLMs and LLMs.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
from itertools import combinations

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

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

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]
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
    ax.set_ylabel("Average gain over standard probing", fontsize=FONTSIZE)
    ax.set_title(f"Dataset difficulty vs gain\nSpearman ρ = {rho:.2f}, p = {pval:.2e}", fontsize=FONTSIZE)
    sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    plt.legend()
    plt.tight_layout()
    if save:
        figname = f"dataset_difficulty_vision_{filename}"
        save_plot(figname=figname)


def plot_pretraining_scale_vision(filename: str, save: bool = False) -> None:
    r"""
    Plot the average gain over standard probing for different pretraining scales.

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

    # Pretraining scale (annotation only, NOT axis geometry)
    pretraining_tokens = {
        "vit": 2.5e11,
        "dinov2": 1.80e12,
        "aimv2": 3.15e12,
        "dinov3": 3.91e12,
    }

    token_labels = {
        "vit": "250B",
        "dinov2": "1.8T",
        "aimv2": "3.2T",
        "dinov3": "3.9T",
    }

    df["pretraining_tokens"] = df["model_name"].map(pretraining_tokens)

    # Order by scale
    model_order = [
        "vit",
        "dinov2",
        "aimv2",
        "dinov3",
    ]

    # Figure
    figsize = 2 * WIDTH, 2 * HEIGHT
    fig = plt.figure(figsize=figsize)

    # ── Median + IQR per model across datasets ────────────────────────────────
    stats = {}
    for m in model_order:
        vals = df[df["model_name"] == m]["gain"].values
        stats[m] = {
            "median": np.median(vals),
            "q25": np.percentile(vals, 25),
            "q75": np.percentile(vals, 75),
        }

    x = np.arange(len(model_order))
    medians = [stats[m]["median"] for m in model_order]
    q25s = [stats[m]["q25"] for m in model_order]
    q75s = [stats[m]["q75"] for m in model_order]

    # Dashed line connecting medians
    plt.plot(
        x,
        medians,
        color="#2D3A8C",
        lw=1.8,
        zorder=3,
        linestyle="--",
        alpha=0.5,
    )

    # IQR band
    plt.fill_between(
        x,
        q25s,
        q75s,
        color="#2D3A8C",
        alpha=0.15,
        zorder=2,
        label="IQR across datasets (Q1–Q3)",
    )

    # One coloured marker per model family
    for xi, m in zip(x, model_order):
        plt.scatter(
            xi,
            stats[m]["median"],
            color=PALETTE[m],
            s=100,
            zorder=4,
            edgecolors="white",
            linewidths=1.2,
            label=m,
        )

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax = fig.axes[0]
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{m}\n{token_labels[m]}" for m in model_order],
        fontsize=FONTSIZE,
    )
    ax.set_ylabel("Median gain over standard probing (pp)", fontsize=FONTSIZE)
    ax.set_xlabel("Model family  /  Pretraining scale", fontsize=FONTSIZE)
    ax.set_xlim(-0.4, len(model_order) - 0.6)
    ax.set_ylim(bottom=0)

    # ── Cosmetics ─────────────────────────────────────────────────────────────
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)
    sns.despine(fig, ax, trim=True, right=True, offset=10)
    ax.legend(frameon=False, fontsize=FONTSIZE - 1, loc="upper right")

    plt.tight_layout()

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

    # ----------------------------
    # Severity order
    # ----------------------------
    severity_order = [1, 3, 5]

    # ----------------------------
    # Figure
    # ----------------------------
    figsize = 2 * WIDTH, 2 * HEIGHT
    fig = plt.figure(figsize=figsize)

    # ----------------------------
    # Barplot with CI
    # ----------------------------
    ax = sns.barplot(
        data=df,
        x="severity",
        y="gain",
        order=severity_order,
        estimator=np.mean,
        errorbar=("ci", 95),
        capsize=0.15,
        color="steelblue",
    )

    # ----------------------------
    # Visualization styling
    # ----------------------------
    ax.grid(alpha=ALPHA_GRID, lw=1.3)
    ax.spines["left"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["top"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", direction="out", length=5, width=1)

    ax.set_xlabel("Corruption severity", fontsize=FONTSIZE)
    ax.set_ylabel("Average gain over standard probing", fontsize=FONTSIZE)

    sns.despine(fig, ax, trim=True, right=True, offset=10)

    plt.tight_layout()

    if save:
        save_plot(figname=f"cifar10c_severity_{filename}")


def analyze_model_size(filename: str) -> None:
    # Recover results
    path = RESULT_DIR / "analysis"
    df = pd.read_csv(path / f"{filename}.csv")
    df = df.copy()

    # Keep vision models
    pattern = "|".join(VFMs)
    df = df[df["model_name"].str.contains(pattern, case=False, na=False)]

    for family, sizes in MODEL_SIZES.items():
        sub = df[df["model_name"] == family]

        # Build matrix: rows=datasets, cols=sizes
        pivot = (
            sub.pivot_table(index="dataset_name", columns="model_size", values="gain").reindex(columns=sizes).dropna()
        )

        print(f"\n{'=' * 50}")
        print(f"Family: {family}  |  sizes: {sizes}  |  datasets: {len(pivot)}")
        print(pivot.round(4).to_string())

        # --- Friedman test ---
        groups = [pivot[s].values for s in pivot.columns]
        stat, p = friedmanchisquare(*groups)

        # Kendall's W (effect size)
        k = len(pivot.columns)
        n = len(pivot)
        W = stat / (k * (n - 1))  # = chi2 / (k*(n-1)) for Friedman

        print(f"\nFriedman: chi2={stat:.3f}, p={p:.4f}, W={W:.3f}")

        if p < 0.05:
            print("→ Significant. Post-hoc pairwise Wilcoxon (Holm-corrected):")
            pairs = list(combinations(pivot.columns, 2))
            raw_p = []
            for s1, s2 in pairs:
                _, pw = wilcoxon(pivot[s1], pivot[s2])
                raw_p.append(pw)
            # Holm correction
            order = np.argsort(raw_p)
            corrected = np.zeros(len(raw_p))
            for rank, idx in enumerate(order):
                corrected[idx] = min(raw_p[idx] * (len(raw_p) - rank), 1.0)
            for (s1, s2), rp, cp in zip(pairs, raw_p, corrected):
                sig = "*" if cp < 0.05 else ""
                print(f"  {s1} vs {s2}: p_raw={rp:.4f}, p_holm={cp:.4f} {sig}")
        else:
            print("→ Not significant. Size does not systematically affect gain.")

        # --- Direction: Spearman between size rank and mean gain ---
        size_ranks = np.arange(len(sizes))  # ordinal: 0=smallest
        mean_gains = [pivot[s].mean() for s in sizes]
        rho, p_rho = spearmanr(size_ranks, mean_gains)
        print(f"Spearman (size rank vs mean gain): rho={rho:.3f}, p={p_rho:.4f}")
        print(f"Mean gain per size: { {s: f'{g:.4f}' for s, g in zip(sizes, mean_gains)} }")


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
    # 1) Frequency of best module
    # ============================================================

    # Best location for every (dataset, model, size)
    idx = df.groupby(["dataset_name", "model_name", "model_size"])["accuracy"].idxmax()

    best = df.loc[idx].copy()

    freq = best["module"].value_counts(normalize=True).rename_axis("module").reset_index(name="frequency")

    freq["frequency"] *= 100

    # Plot
    plt.figure(figsize=(5, 3.5))
    sns.barplot(
        data=freq,
        x="module",
        y="frequency",
        color="steelblue",
    )

    plt.ylabel("Winning frequency over 150 configurations (%)")
    plt.xlabel("")
    plt.grid(alpha=0.3)
    sns.despine()
    plt.tight_layout()

    print(freq)

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
    # Overall table
    # ============================================================

    overall = summary.groupby("strategy").agg(
        gain=("gain", "mean"),
        recovered=("recovered", "mean"),
        optimal_frequency=("exact", "mean"),
    )

    overall["optimal_frequency"] *= 100

    print("\n================ OVERALL ================\n")
    print(overall.round(2))

    # ============================================================
    # Per dataset
    # ============================================================

    per_dataset = summary.groupby(["dataset", "strategy"]).agg(
        gain=("gain", "median"),
        recovered=("recovered", "mean"),
        optimal_frequency=("exact", "mean"),
    )

    per_dataset["optimal_frequency"] *= 100

    print("\n================ DATASET ================\n")
    print(per_dataset.round(2))

    # ============================================================
    # Per family
    # ============================================================

    per_family = summary.groupby(["family", "strategy"]).agg(
        gain=("gain", "median"),
        recovered=("recovered", "mean"),
        optimal_frequency=("exact", "mean"),
    )

    per_family["optimal_frequency"] *= 100

    print("\n================ FAMILY ================\n")
    print(per_family.round(2))


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


# ------------------------------------------------------------------------------
# Results functions
# ------------------------------------------------------------------------------
def get_tables() -> None:

    filename = "cls"
    analyze_model_size(filename=filename)
    analyze_module(filename=filename)


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

    # # Pretraining scale
    # save = True
    # filename = "cls"
    # plot_pretraining_scale_vision(filename=filename, save=save)

    # # Model family
    # save = True
    # filename = "cls"
    # plot_model_family_vision(filename=filename, save=save)

    # Cifar10-C
    save = True
    filename = "cifar10_c"
    plot_cifar10c_severity(filename=filename, save=save)


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
