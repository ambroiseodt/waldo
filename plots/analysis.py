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
import pandas as pd

from waldo.infra import FIGURE_DIR, RESULT_DIR

logger = logging.getLogger("waldo.analysis")


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


save = True
model_name = "clip"
model_size = "large"
patch_size = 14
resolution = 224
dataset_name = "food101"
plot_probe_comparison_image_classification(
    dataset_name=dataset_name,
    model_name=model_name,
    model_size=model_size,
    patch_size=patch_size,
    resolution=resolution,
    save=save,
)
