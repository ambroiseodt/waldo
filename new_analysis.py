"""Plot oracle-winning probing-location distributions for VFMs and LLMs.

The main figure is a 2-D density map: each cell contains the fraction of
evaluation units for which that layer/module location is oracle-optimal. Ties
are split evenly, so the map has total mass one. The lower panels show global
winning-frequency slices at the selected layer and on RC2.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

MODULE_ORDER = ["ln1", "mha", "rc1", "ln2", "fc1", "act", "fc2", "rc2"]
MODULE_LABELS = {module: ("Act" if module == "act" else module.upper()) for module in MODULE_ORDER}
VFM_NAMES = {"vit", "dinov2", "aimv2", "dinov3"}
LLM_NAMES = {
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Llama-3.2-1B",
    "Llama-3.2-3B",
    "Llama-3.1-8B",
    "gemma-3-1b-pt",
    "gemma-3-4b-pt",
    "gemma-3-12b-pt",
}
MODULE_COLORS = {"act": "#96c0cf", "ln2": "#b153a1", "rc2": "#428379"}
STANDARD_LOCATION = (100.0, "rc2")
FIXED_LOCATIONS = {"vision": (100.0, "act"), "language": (60.0, "ln2")}


def _winner_data(filename: str, modality: str) -> tuple[pd.DataFrame, list[str], str, float]:
    """Load raw scores and return tie-aware, weighted oracle winners.

    Language scores are first averaged over tasks for each model/location,
    matching the MTEB-level analysis. Vision evaluation units are individual
    dataset/model/size triples. The raw grid is used instead of the summary
    table because the latter drops ties and non-winning locations.
    """
    df = pd.read_csv(RESULTS / filename)

    if modality == "language":
        df = df.groupby(["model_name", "layer", "module"], as_index=False, observed=True)["accuracy"].mean()
        df = df[df["model_name"].isin(LLM_NAMES)].copy()
        group_cols = ["model_name"]
    elif modality == "vision":
        df = df[df["model_name"].str.lower().isin(VFM_NAMES)].copy()
        group_cols = ["dataset_name", "model_name", "model_size"]
    else:
        raise ValueError("modality must be 'vision' or 'language'")

    if df.empty:
        raise ValueError(f"No {modality} rows found in {filename}")

    grouped = df.groupby(group_cols, observed=True)
    last_layer = grouped["layer"].transform("max")
    oracle_accuracy = grouped["accuracy"].transform("max")
    is_winner = np.isclose(df["accuracy"], oracle_accuracy, atol=1e-8, rtol=0)
    winners = df.loc[is_winner].copy()
    winners["last_layer"] = last_layer.loc[winners.index]
    winners["depth_pct"] = 100 * winners["layer"] / winners["last_layer"].clip(lower=1)

    # A tied location receives part of one unit of mass. Consequently the
    # total of weights is exactly the number of evaluation units.
    winners["weight"] = 1 / winners.groupby(group_cols, observed=True)["module"].transform("size")
    winners.attrs["n_units"] = int(grouped.ngroups)
    winners.attrs["group_cols"] = group_cols
    fixed_depth, fixed_module = FIXED_LOCATIONS[modality]
    return winners, group_cols, fixed_module, fixed_depth


def _density_grid(winners: pd.DataFrame, n_units: int, depth_bin: int) -> tuple[np.ndarray, np.ndarray]:
    """Bin winner mass by relative depth and module without interpolation."""
    depth_edges = np.arange(0, 100 + depth_bin, depth_bin, dtype=float)
    bin_index = np.digitize(winners["depth_pct"].to_numpy(), depth_edges, right=False) - 1
    bin_index = np.clip(bin_index, 0, len(depth_edges) - 2)
    module_index = winners["module"].map({module: i for i, module in enumerate(MODULE_ORDER)}).to_numpy()
    grid = np.zeros((len(MODULE_ORDER), len(depth_edges) - 1), dtype=float)
    np.add.at(grid, (module_index, bin_index), winners["weight"].to_numpy() / n_units)

    return grid, depth_edges


def _slice_frequency(winners: pd.DataFrame, mask: pd.Series, n_units: int) -> np.ndarray:
    """Return the global winning frequency in one section of the grid."""
    selected = winners.loc[mask]
    values = selected.groupby("module", observed=True)["weight"].sum().reindex(MODULE_ORDER, fill_value=0)
    return (values / n_units).to_numpy()


def plot_winner_distribution(filename: str, modality: str, depth_bin: int = 5) -> None:
    """Write a heatmap, two diagnostic slices, and a sparse 3-D bar plot."""
    winners, _, fixed_module, fixed_depth = _winner_data(filename, modality)
    n_units = winners.attrs["n_units"]
    grid, depth_edges = _density_grid(winners, n_units, depth_bin)
    depth_left = depth_edges[:-1]
    depth_centers = depth_left + depth_bin / 2

    layer_slice = 100.0 if modality == "vision" else fixed_depth
    layer_mask = np.isclose(winners["depth_pct"], layer_slice)
    layer_freq = _slice_frequency(winners, layer_mask, n_units)
    rc2_mask = winners["module"].eq("rc2")
    rc2_mass = winners.loc[rc2_mask].groupby("depth_pct", observed=True)["weight"].sum()
    rc2_freq = np.zeros(len(depth_centers), dtype=float)
    rc2_indices = np.digitize(rc2_mass.index.to_numpy(), depth_edges, right=False) - 1
    rc2_indices = np.clip(rc2_indices, 0, len(rc2_freq) - 1)
    np.add.at(rc2_freq, rc2_indices, rc2_mass.to_numpy() / n_units)

    label = "VFM" if modality == "vision" else "LLM"
    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin=0, vmax=max(float(grid.max()), 1e-12))
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    axes = fig.subplot_mosaic([["density", "density"], ["final", "rc2"]], height_ratios=[2.1, 1.2])
    ax_heat, ax_final, ax_rc2 = axes["density"], axes["final"], axes["rc2"]

    image = ax_heat.pcolormesh(
        depth_edges, np.arange(len(MODULE_ORDER) + 1) - 0.5, grid, cmap=cmap, norm=norm, shading="flat"
    )
    ax_heat.axvline(STANDARD_LOCATION[0], color="white", linestyle="--", linewidth=1.5, zorder=3)
    ax_heat.scatter(
        STANDARD_LOCATION[0],
        MODULE_ORDER.index(STANDARD_LOCATION[1]),
        marker="s",
        s=90,
        facecolors="none",
        edgecolors="white",
        linewidths=2,
        zorder=4,
    )
    ax_heat.scatter(
        fixed_depth,
        MODULE_ORDER.index(fixed_module),
        marker="*",
        s=180,
        facecolors="none",
        edgecolors="#54c7ff",
        linewidths=1.5,
        zorder=4,
    )
    ax_heat.set(
        xlim=(0, 102),
        ylim=(-0.5, len(MODULE_ORDER) - 0.5),
        xlabel="Relative layer depth (%)",
        ylabel="Module",
        yticks=np.arange(len(MODULE_ORDER)),
        yticklabels=[MODULE_LABELS[module] for module in MODULE_ORDER],
        title=f"{label}: oracle-winning location density ({n_units} evaluation units)",
    )
    ax_heat.grid(False)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="white",
            markerfacecolor="none",
            markersize=8,
            linewidth=1.5,
            linestyle="--",
            label="Standard: final + RC2",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="#54c7ff",
            markerfacecolor="none",
            markersize=11,
            linewidth=0,
            label=f"Fixed alternative: {MODULE_LABELS[fixed_module]} at {fixed_depth:.0f}%",
        ),
    ]
    ax_heat.legend(handles=legend_handles, loc="upper left", fontsize=9, frameon=True)
    fig.colorbar(image, ax=ax_heat, fraction=0.025, pad=0.015, label="Winning frequency per evaluation unit")

    positions = np.arange(len(MODULE_ORDER))
    ax_final.bar(
        positions,
        layer_freq,
        color=[MODULE_COLORS.get(module, "#777777") for module in MODULE_ORDER],
        edgecolor="#333333",
        linewidth=0.35,
    )
    if layer_slice == 100:
        ax_final.scatter(
            MODULE_ORDER.index("rc2"),
            layer_freq[MODULE_ORDER.index("rc2")],
            marker="s",
            s=55,
            facecolors="none",
            edgecolors="#222222",
            zorder=3,
        )
    if fixed_depth == layer_slice:
        ax_final.scatter(
            MODULE_ORDER.index(fixed_module),
            layer_freq[MODULE_ORDER.index(fixed_module)],
            marker="*",
            s=100,
            facecolors="none",
            edgecolors="#0077aa",
            zorder=3,
        )
    ax_final.set(
        title=f"Slice: layer = {layer_slice:.0f}%",
        xlabel="Module",
        ylabel="Winning frequency",
        xticks=positions,
        xticklabels=[MODULE_LABELS[module] for module in MODULE_ORDER],
        ylim=(0, max(layer_freq.max() * 1.18, 0.1)),
    )
    ax_final.tick_params(axis="x", rotation=50)
    if not layer_freq.any():
        ax_final.text(
            0.5, 0.5, "No oracle winners", transform=ax_final.transAxes, ha="center", va="center", fontsize=10
        )

    ax_rc2.bar(
        depth_centers, rc2_freq, width=depth_bin * 0.9, color=MODULE_COLORS["rc2"], edgecolor="#333333", linewidth=0.35
    )
    ax_rc2.axvline(100, color="#222222", linestyle="--", linewidth=1.2)
    if fixed_module == "rc2":
        ax_rc2.axvline(fixed_depth, color="#0077aa", linestyle=":", linewidth=1.5)
    ax_rc2.set(
        title="Slice: module = RC2",
        xlabel="Relative layer depth (%)",
        ylabel="Winning frequency",
        xlim=(0, 102),
        ylim=(0, max(rc2_freq.max() * 1.18, 0.1)),
    )
    ax_rc2.set_xticks(np.arange(0, 101, 20))
    if not rc2_freq.any():
        ax_rc2.text(0.5, 0.5, "No oracle winners", transform=ax_rc2.transAxes, ha="center", va="center", fontsize=10)

    for axis in (ax_final, ax_rc2):
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    stem = f"oracle_winner_intro_{modality}_{filename.removesuffix('.csv')}"
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    # A sparse bar rendering avoids inventing sloped surfaces between the
    # discrete module/depth cells. The heatmap remains the recommended view.
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    rows, cols = np.nonzero(grid > 0)
    colors = cmap(norm(grid[rows, cols]))
    ax.bar3d(
        depth_left[cols],
        rows - 0.35,
        np.zeros_like(rows, dtype=float),
        depth_bin * 0.85,
        0.7,
        grid[rows, cols],
        color=colors,
        shade=False,
    )
    ax.set(
        xlabel="Relative layer depth (%)",
        ylabel="Module",
        zlabel="Winning frequency",
        title=f"{label}: sparse oracle-winning density",
        yticks=np.arange(len(MODULE_ORDER)),
        yticklabels=[MODULE_LABELS[module] for module in MODULE_ORDER],
        xlim=(0, 102),
    )
    ax.view_init(elev=28, azim=-60)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.65,
        pad=0.12,
        label="Winning frequency per evaluation unit",
    )
    fig.subplots_adjust(left=0.0, right=0.85, bottom=0.05, top=0.92)
    fig.savefig(FIGURES / f"{stem}_3d.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"{label}: {n_units} evaluation units; wrote {stem}.pdf and {stem}_3d.pdf")


def plot_bubble_distribution(filename: str, modality: str, depth_bin: int = 5) -> None:
    """Save a sparse bubble matrix alternative to the intro heatmap."""
    winners, _, fixed_module, fixed_depth = _winner_data(filename, modality)
    n_units = winners.attrs["n_units"]
    grid, depth_edges = _density_grid(winners, n_units, depth_bin)
    rows, cols = np.nonzero(grid > 0)
    frequencies = grid[rows, cols]
    depth_centers = depth_edges[:-1] + depth_bin / 2
    label = "VFM" if modality == "vision" else "LLM"
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    sizes = 80 + 1800 * frequencies / max(frequencies.max(), 1e-12)
    points = ax.scatter(
        depth_centers[cols], rows, s=sizes, c=frequencies, cmap="magma", edgecolors="#333333", linewidths=0.4
    )
    ax.set(
        xlim=(0, 102),
        ylim=(-0.5, len(MODULE_ORDER) - 0.5),
        xlabel="Relative layer depth (%)",
        ylabel="Module",
        title=f"{label}: oracle-winning locations (bubble view)",
        yticks=np.arange(len(MODULE_ORDER)),
        yticklabels=[MODULE_LABELS[module] for module in MODULE_ORDER],
    )
    ax.axvline(STANDARD_LOCATION[0], color="#333333", linestyle="--", linewidth=1.2)
    ax.scatter(
        fixed_depth,
        MODULE_ORDER.index(fixed_module),
        marker="*",
        s=180,
        facecolors="none",
        edgecolors="#0077aa",
        linewidths=1.5,
        zorder=3,
    )
    fig.colorbar(points, ax=ax, label="Winning frequency per evaluation unit")
    fig.savefig(FIGURES / f"oracle_winner_bubble_{modality}_{filename.removesuffix('.csv')}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_contour_distribution(filename: str, modality: str, depth_bin: int = 5) -> None:
    """Save a smoothed contour view for exploration, not primary reporting."""
    from scipy.ndimage import gaussian_filter

    winners, _, fixed_module, fixed_depth = _winner_data(filename, modality)
    n_units = winners.attrs["n_units"]
    grid, depth_edges = _density_grid(winners, n_units, depth_bin)
    smoothed = gaussian_filter(grid, sigma=(0.45, 1.25), mode="nearest")
    label = "VFM" if modality == "vision" else "LLM"
    x = depth_edges[:-1] + depth_bin / 2
    y = np.arange(len(MODULE_ORDER))
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    levels = np.linspace(0, max(float(smoothed.max()), 1e-12), 8)[1:]
    filled = ax.contourf(x, y, smoothed, levels=levels, cmap="magma", extend="max")
    ax.contour(x, y, smoothed, levels=levels, colors="#eeeeee", linewidths=0.45, alpha=0.7)
    ax.set(
        xlim=(0, 102),
        ylim=(-0.5, len(MODULE_ORDER) - 0.5),
        xlabel="Relative layer depth (%)",
        ylabel="Module",
        title=f"{label}: smoothed oracle-winning density (exploratory)",
        yticks=y,
        yticklabels=[MODULE_LABELS[module] for module in MODULE_ORDER],
    )
    ax.axvline(STANDARD_LOCATION[0], color="white", linestyle="--", linewidth=1.2)
    ax.scatter(
        fixed_depth,
        MODULE_ORDER.index(fixed_module),
        marker="*",
        s=180,
        facecolors="none",
        edgecolors="#54c7ff",
        linewidths=1.5,
        zorder=3,
    )
    fig.colorbar(filled, ax=ax, label="Smoothed winning frequency")
    fig.savefig(FIGURES / f"oracle_winner_contour_{modality}_{filename.removesuffix('.csv')}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_winner_distribution("cls.csv", "vision")
    plot_winner_distribution("mteb.csv", "language")
    plot_bubble_distribution("cls.csv", "vision")
    plot_bubble_distribution("mteb.csv", "language")
    plot_contour_distribution("cls.csv", "vision")
    plot_contour_distribution("mteb.csv", "language")
