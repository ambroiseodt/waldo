r"""
Analyze evaluation results from VFMs, VLMs and LLMs.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from waldo.infra import FIGURE_DIR, RESULT_DIR

logger = logging.getLogger("waldo.analysis")

MODULE_ORDER = {
    "ln1": 0,
    "mha": 1,
    "rc1": 2,
    "ln2": 3,
    "fc1": 4,
    "act": 5,
    "fc2": 6,
    "rc2": 7,
}

MODEL_PARAMS = {
    ("aimv2", "1B"): 1e9,
    ("aimv2", "3B"): 3e9,
    ("aimv2", "large"): 3e8,
    ("aimv2", "huge"): 6e8,
    ("dinov2", "small"): 2e7,
    ("dinov2", "base"): 8e7,
    ("dinov2", "large"): 3e8,
    ("dinov2", "giant"): 1e8,
    ("dinov3", "vits16"): 2e7,
    ("dinov3", "vitb16"): 8e7,
    ("dinov3", "vitl16"): 3e8,
    ("dinov3", "vit7b16"): 7e9,
    ("vit", "base"): 8e7,
    ("vit", "large"): 3e8,
    ("vit", "huge"): 6e8,
    ("clip", "base"): 8e7,
    ("clip", "large"): 3e8,
    ("siglip2", "base"): 8e7,
    ("siglip2", "large"): 3e8,
    ("siglip2", "giant-opt"): 1e9,
    # Complete with the remaining models
}


def save_plot(figname: str, format: str = "pdf", dpi: int = 100) -> None:
    """Save figure in pdf format."""
    figure_path = FIGURE_DIR
    if not figure_path.exists():
        figure_path.mkdir(parents=True, exist_ok=True)
    save_dir = figure_path / f"{figname}.{format}"
    plt.savefig(save_dir, format=format, bbox_inches="tight", dpi=dpi)


def plot_probe_comparison_image_classification(
    dataset_name, model_name, model_size, patch_size, resolution, save=False
):
    path = RESULT_DIR / "analysis" / "image_classification.csv"
    df = pd.read_csv(path)

    # filter one experiment
    row = df[
        (df["model_name"] == model_name)
        & (df["model_size"] == model_size)
        & (df["patch_size"] == patch_size)
        & (df["resolution"] == resolution)
        & (df["dataset_name"] == dataset_name)
    ]

    if row.empty:
        raise ValueError("No matching experiment found.")

    row = row.iloc[0]

    labels = ["Standard (last, rc2)", "Best layer (rc2 fixed)", "Best module (last layer fixed)", "Global best"]

    values = [
        row["baseline_accuracy"],
        row["best_layer_rc2_accuracy"],
        row["best_module_last_layer_accuracy"],
        row["best_accuracy"],
    ]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values)

    plt.ylabel("Accuracy")

    title = f"{model_name} | {model_size} | {dataset_name}"
    plt.title(title)

    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save:
        figname = f"probe_comparison_{model_name}_{model_size}_{dataset_name}"
        save_plot(figname=figname)
    plt.show()


def plot_best_locations_mteb(save: bool = False) -> None:
    """Plot optimal probing locations for a given dataset.

    x-axis: layer depth (%)
    y-axis: module position in transformer block
    marker size: model size (#params)
    color: training paradigm
    """
    path = RESULT_DIR / "analysis" / "mteb.csv"
    df_data = pd.read_csv(path)
    task_names = df_data["task_name"].unique()
    for task_name in task_names:
        df = df_data[(df_data["task_name"] == task_name)].copy()

        if df.empty:
            print(f"No results found for task '{task_name}'.")
            return

        df["best_layer_pct"] = 100 * df["best_layer"] / (df["n_layers"] - 1)
        df["module_pos"] = df["best_module"].map(MODULE_ORDER)

        fig, ax = plt.subplots(figsize=(10, 6))

        for _, row in df.iterrows():
            marker_size = 50

            ax.scatter(
                row["best_layer_pct"],
                row["module_pos"],
                s=marker_size,
                alpha=0.75,
                edgecolor="black",
                linewidth=0.5,
            )

            ax.annotate(
                f"{row['model_name']}",
                (row["best_layer_pct"], row["module_pos"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

        ax.set_xlim(-2, 102)

        ax.set_yticks(range(len(MODULE_ORDER)))
        ax.set_yticklabels(list(MODULE_ORDER.keys()))

        ax.set_xlabel("Layer Depth (%)")
        ax.set_ylabel("Transformer Module")
        ax.set_title(f"Optimal Probing Locations — {task_name}")

        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        if save:
            figname = f"best_locations_mteb_{task_name}"
            save_plot(figname=figname)
        plt.close()


def plot_best_locations(dataset_name: str, save: bool = False) -> None:
    """Plot optimal probing locations for a given dataset.

    x-axis: layer depth (%)
    y-axis: module position in transformer block
    marker size: model size (#params)
    color: training paradigm
    """
    path = RESULT_DIR / "analysis" / "image_classification.csv"
    df = pd.read_csv(path)

    df = df[df["dataset_name"] == dataset_name].copy()

    if df.empty:
        print(f"No results found for dataset '{dataset_name}'.")
        return

    df["best_layer_pct"] = 100 * df["best_layer"] / (df["n_layers"] - 1)
    df["module_pos"] = df["best_module"].map(MODULE_ORDER)

    color_map = {
        "autoregressive": "tab:red",
        "ssl_distillation": "tab:green",
        "contrastive": "tab:blue",
        "supervised": "tab:orange",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in df.iterrows():
        params = MODEL_PARAMS.get(
            (row["model_name"], row["model_size"]),
            1e8,
        )

        marker_size = 50 + 450 * np.sqrt(params / 1e9)

        ax.scatter(
            row["best_layer_pct"],
            row["module_pos"],
            s=marker_size,
            color=color_map.get(row["training_paradigm"], "gray"),
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.annotate(
            f"{row['model_name']}-{row['model_size']}",
            (row["best_layer_pct"], row["module_pos"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlim(-2, 102)

    ax.set_yticks(range(len(MODULE_ORDER)))
    ax.set_yticklabels(list(MODULE_ORDER.keys()))

    ax.set_xlabel("Layer Depth (%)")
    ax.set_ylabel("Transformer Module")
    ax.set_title(f"Optimal Probing Locations — {dataset_name}")

    ax.grid(True, linestyle="--", alpha=0.4)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color,
            label=paradigm,
            markersize=10,
        )
        for paradigm, color in color_map.items()
    ]

    ax.legend(
        handles=legend_handles,
        title="Training paradigm",
        loc="upper left",
    )

    plt.tight_layout()
    if save:
        figname = f"best_locations_{dataset_name}"
        save_plot(figname=figname)
    plt.close()


def plot_best_locations_domainnet(domain: str, save: bool = False) -> None:
    """Plot optimal probing locations for a given dataset.

    x-axis: layer depth (%)
    y-axis: module position in transformer block
    marker size: model size (#params)
    color: training paradigm
    """
    path = RESULT_DIR / "analysis" / "domainnet.csv"
    df = pd.read_csv(path)

    df = df[df["domain"] == domain].copy()

    if df.empty:
        print(f"No results found for domain '{domain}'.")
        return

    df["best_layer_pct"] = 100 * df["best_layer"] / (df["n_layers"] - 1)
    df["module_pos"] = df["best_module"].map(MODULE_ORDER)

    color_map = {
        "autoregressive": "tab:red",
        "ssl_distillation": "tab:green",
        "contrastive": "tab:blue",
        "supervised": "tab:orange",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in df.iterrows():
        params = MODEL_PARAMS.get(
            (row["model_name"], row["model_size"]),
            1e8,
        )

        marker_size = 50 + 450 * np.sqrt(params / 1e9)

        ax.scatter(
            row["best_layer_pct"],
            row["module_pos"],
            s=marker_size,
            color=color_map.get(row["training_paradigm"], "gray"),
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.annotate(
            f"{row['model_name']}-{row['model_size']}",
            (row["best_layer_pct"], row["module_pos"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlim(-2, 102)

    ax.set_yticks(range(len(MODULE_ORDER)))
    ax.set_yticklabels(list(MODULE_ORDER.keys()))

    ax.set_xlabel("Layer Depth (%)")
    ax.set_ylabel("Transformer Module")
    ax.set_title(f"Optimal Probing Locations — {domain}")

    ax.grid(True, linestyle="--", alpha=0.4)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color,
            label=paradigm,
            markersize=10,
        )
        for paradigm, color in color_map.items()
    ]

    ax.legend(
        handles=legend_handles,
        title="Training paradigm",
        loc="upper left",
    )

    plt.tight_layout()
    if save:
        figname = f"best_locations_domainnet_{domain}"
        save_plot(figname=figname)
    plt.close()


def plot_best_locations_cifar10_c(corruption: str, severity: int, save: bool = False) -> None:
    """Plot optimal probing locations for a given dataset.

    x-axis: layer depth (%)
    y-axis: module position in transformer block
    marker size: model size (#params)
    color: training paradigm
    """
    path = RESULT_DIR / "analysis" / "cifar10_c.csv"
    df = pd.read_csv(path)

    df = df[(df["corruption"] == corruption) & (df["severity"] == severity)].copy()

    if df.empty:
        print(f"No results found for corruption '{corruption}' and severity '{severity}'.")
        return

    df["best_layer_pct"] = 100 * df["best_layer"] / (df["n_layers"] - 1)
    df["module_pos"] = df["best_module"].map(MODULE_ORDER)

    color_map = {
        "autoregressive": "tab:red",
        "ssl_distillation": "tab:green",
        "contrastive": "tab:blue",
        "supervised": "tab:orange",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in df.iterrows():
        params = MODEL_PARAMS.get(
            (row["model_name"], row["model_size"]),
            1e8,
        )

        marker_size = 50 + 450 * np.sqrt(params / 1e9)

        ax.scatter(
            row["best_layer_pct"],
            row["module_pos"],
            s=marker_size,
            color=color_map.get(row["training_paradigm"], "gray"),
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.annotate(
            f"{row['model_name']}-{row['model_size']}",
            (row["best_layer_pct"], row["module_pos"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlim(-2, 102)

    ax.set_yticks(range(len(MODULE_ORDER)))
    ax.set_yticklabels(list(MODULE_ORDER.keys()))

    ax.set_xlabel("Layer Depth (%)")
    ax.set_ylabel("Transformer Module")
    ax.set_title(f"Optimal Probing Locations — {corruption} - {severity}")

    ax.grid(True, linestyle="--", alpha=0.4)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color,
            label=paradigm,
            markersize=10,
        )
        for paradigm, color in color_map.items()
    ]

    ax.legend(
        handles=legend_handles,
        title="Training paradigm",
        loc="upper left",
    )

    plt.tight_layout()
    if save:
        figname = f"best_locations_cifar10_c_{corruption}_{severity}"
        save_plot(figname=figname)
    plt.close()


# save = True
# model_name = "clip"
# model_size = "large"
# patch_size = 14
# resolution = 224
# dataset_name = "food101"
# plot_probe_comparison_image_classification(
#     dataset_name=dataset_name,
#     model_name=model_name,
#     model_size=model_size,
#     patch_size=patch_size,
#     resolution=resolution,
#     save=save,
# )

save = True
plot_best_locations_mteb(save=save)

# save = True
# for dataset_name in [
#     "aircraft",
#     "cifar10",
#     "cifar10_c-corruption-motion_blur-severity-5",
#     "cifar100",
#     "domainnet-sketch",
#     "dtd",
#     "eurosat",
#     "flowers102",
#     "food101",
#     "pet",
# ]:
#     plot_best_locations(dataset_name=dataset_name, save=save)

# save = True
# for domain in ["clipart", "infograph", "painting", "quickdraw", "sketch"]:
#     plot_best_locations_domainnet(domain=domain, save=save)

# save = True
# for corruption in ["contrast", "gaussian_noise", "motion_blur", "snow", "speckle_noise"]:
#     for severity in [1, 3, 5]:
#         plot_best_locations_cifar10_c(corruption=corruption, severity=severity, save=save)
