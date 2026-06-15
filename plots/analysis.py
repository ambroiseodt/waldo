r"""
Analyze evaluation results from VFMs, VLMs and LLMs.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from waldo.infra import RESULT_DIR

logger = logging.getLogger("waldo.analysis")


def plot_global_representation_profile(model, output_dir="figures"):
    """Generates and saves a 4x6 grid of line plots for all tasks,

    showing mean accuracy (%) and std variance vs layer depth (%) for each module.
    """
    path = RESULT_DIR / "mteb.csv"  # Ensure RESULT_DIR is defined globally
    df = pd.read_csv(path)

    # 1. Filter for the specific model
    model_df = df[df["model_name"] == model].copy()

    if model_df.empty:
        print(f"Error: No data found for model '{model}'")
        return

    # 2. Dynamically extract all unique tasks present in the data
    unique_tasks = sorted(model_df["task_name"].unique())
    num_tasks = len(unique_tasks)

    # 3. Process metrics and scale layer depth to percentages
    normalized_dfs = []
    for task in unique_tasks:
        task_df = model_df[model_df["task_name"] == task].copy()

        # Convert accuracy to percentage scale (0-100)
        task_df["accuracy_pct"] = task_df["accuracy"] * 100

        # Convert layer depth to percentage: (current_layer / max_layer) * 100
        max_layer = task_df["layer"].max()
        if max_layer > 0:
            task_df["layer_pct"] = (task_df["layer"] / max_layer) * 100
        else:
            task_df["layer_pct"] = 0.0

        normalized_dfs.append(task_df)

    # Recombine the processed data rows
    model_df = pd.concat(normalized_dfs)

    # 4. Set up the 4x6 Grid Subplot Layout
    ncols = 6
    nrows = math.ceil(num_tasks / ncols)

    # sharey=True keeps the Y-axis scaled identically across all tasks for easy comparison.
    # Change to sharey=False if individual datasets have vastly different baseline scales.
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 16))  # , sharex=True, sharey=True)
    axes = axes.flatten()

    # Track handles and labels for the global legend
    legend_dict = {}

    # 5. Populate each subplot with curves and variance bands
    for i, task in enumerate(unique_tasks):
        ax = axes[i]

        # Isolate current task data
        task_data = model_df[model_df["task_name"] == task]

        # Compute Mean and Std across the 10 experimental seeds/IDs
        # grouping by module and layer percentage
        grouped = task_data.groupby(["module", "layer_pct"])["accuracy_pct"].agg(["mean", "std"]).reset_index()

        # Group by module to plot a separate line/band for each
        for module_name, module_df in grouped.groupby("module"):
            # Sort by layer percentage to ensure lines draw smoothly left-to-right
            module_df = module_df.sort_values("layer_pct")

            x = module_df["layer_pct"]
            y_mean = module_df["mean"]
            y_std = module_df["std"].fillna(0)  # Handle cases with 1 run safely

            # Plot the mean performance trajectory
            (line,) = ax.plot(x, y_mean, label=module_name, linewidth=2, alpha=0.9)

            # # Shade the standard deviation area around the mean profile
            # ax.fill_between(
            #     x,
            #     y_mean - y_std,
            #     y_mean + y_std,
            #     color=line.get_color(),
            #     alpha=0.15,
            # )

            # Store the handle for the global legend
            if module_name not in legend_dict:
                legend_dict[module_name] = line

        # Clean up subplot styling
        ax.set_title(task, fontsize=11, weight="bold", pad=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Hide any unused subplots
    for j in range(num_tasks, len(axes)):
        fig.delaxes(axes[j])

    # 6. Add Global Labels and Context
    fig.text(
        0.5,
        0.05,
        "Layer Depth (%)",
        ha="center",
        fontsize=14,
        weight="semibold",
    )
    fig.text(
        0.01,
        0.5,
        "Accuracy (%) [Mean $\\pm$ Std]",
        va="center",
        rotation="vertical",
        fontsize=14,
        weight="semibold",
    )

    # Create a single, clean global legend positioned at the bottom
    sorted_modules = sorted(legend_dict.keys())
    ordered_handles = [legend_dict[m] for m in sorted_modules]

    fig.legend(
        ordered_handles,
        sorted_modules,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(sorted_modules),
        fontsize=12,
        frameon=True,
    )

    # Adjust layout to fit titles and the global legend perfectly
    plt.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.10, hspace=0.30, wspace=0.15)
    fig.suptitle(
        f"Global Representation Analysis Profile: {model}",
        fontsize=20,
        weight="bold",
    )

    # 7. Sanitize name and save the figure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_model_name = model.replace("/", "_").replace(" ", "_")
    output_filepath = Path(output_dir) / f"global_profile_{safe_model_name}.png"

    plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
    print(f"--> Successfully saved global profile plot to: {output_filepath}")

    plt.close(fig)


def plot_aggregated_representation_profile(model, output_dir="figures"):
    """Generates and saves a single line plot showing the model's performance

    averaged across all tasks, with one curve per module across layer depth (%).
    """
    path = RESULT_DIR / "mteb.csv"  # Ensure RESULT_DIR is defined globally
    df = pd.read_csv(path)

    # 1. Filter for the specific model
    model_df = df[df["model_name"] == model].copy()

    if model_df.empty:
        print(f"Error: No data found for model '{model}'")
        return

    # 2. Process metrics and scale layer depth to percentages per task
    unique_tasks = model_df["task_name"].unique()
    processed_dfs = []

    for task in unique_tasks:
        task_df = model_df[model_df["task_name"] == task].copy()

        # Convert accuracy to percentage scale (0-100)
        task_df["accuracy_pct"] = task_df["accuracy"] * 100

        # Convert layer depth to percentage: (current_layer / max_layer) * 100
        max_layer = task_df["layer"].max()
        if max_layer > 0:
            task_df["layer_pct"] = (task_df["layer"] / max_layer) * 100
        else:
            task_df["layer_pct"] = 0.0

        # Step A: Average across the 10 experimental runs *within* this task
        task_grouped = task_df.groupby(["module", "layer_pct"])["accuracy_pct"].mean().reset_index()
        processed_dfs.append(task_grouped)

    # Combine all task averages
    all_tasks_df = pd.concat(processed_dfs)

    # Step B: Average across all unique datasets/tasks
    aggregated_df = all_tasks_df.groupby(["module", "layer_pct"])["accuracy_pct"].mean().reset_index()

    # 3. Set up the single plot canvas
    fig, ax = plt.subplots(figsize=(10, 6))

    # 4. Plot a curve for each module
    for module_name, module_data in aggregated_df.groupby("module"):
        module_data = module_data.sort_values("layer_pct")

        ax.plot(
            module_data["layer_pct"],
            module_data["accuracy_pct"],
            label=module_name,
            linewidth=2.5,
            alpha=0.9,
        )

    # 5. Styling and Labels
    ax.set_title(
        f"Aggregated Representation Profile\nModel: {model} (Averaged over {len(unique_tasks)} Tasks)",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    ax.set_xlabel("Layer Depth (%)", fontsize=12, weight="semibold", labelpad=10)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=12, weight="semibold", labelpad=10)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(-2, 102)  # Margin padding for the edges

    # Place the legend inside or just outside the plot area
    ax.legend(title="Modules", fontsize=10, title_fontsize=11, loc="best")

    plt.tight_layout()

    # 6. Sanitize name and save the figure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_model_name = model.replace("/", "_").replace(" ", "_")
    output_filepath = Path(output_dir) / f"aggregated_profile_{safe_model_name}.png"

    plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
    print(f"--> Successfully saved aggregated profile plot to: {output_filepath}")

    plt.close(fig)


def print_top_modules_per_task(model):
    """Identifies and prints the highest-performing module across layers

    for each task in a clean, tabular format.
    """
    path = RESULT_DIR / "mteb.csv"  # Ensure RESULT_DIR is defined globally
    df = pd.read_csv(path)

    # 1. Filter for the specific model
    model_df = df[df["model_name"] == model].copy()

    if model_df.empty:
        print(f"Error: No data found for model '{model}'")
        return

    # 2. Convert accuracy to percentage (0-100)
    model_df["accuracy_pct"] = model_df["accuracy"] * 100

    # 3. Step A: Average the 10 experimental runs to get a stable score
    # for each unique combinations of task, module, and layer
    run_averages = model_df.groupby(["task_name", "module", "layer"])["accuracy_pct"].mean().reset_index()

    # Step B: Find the absolute peak performance for each module within each task
    # (collapsing the layer dimension to find the module's best moment)
    module_peaks = run_averages.groupby(["task_name", "module"])["accuracy_pct"].max().reset_index()

    # 4. Step C: For each task, find which module hit the highest overall peak
    idx_best = module_peaks.groupby("task_name")["accuracy_pct"].idxmax()
    best_modules = module_peaks.loc[idx_best].sort_values("task_name")

    # 5. Print the results as a cleanly formatted text table
    print(f"\nTop Performing Modules per Task for Model: {model}")
    print("-" * 75)
    # Header with explicit column widths: 40 chars for Task, 18 for Module, 12 for Accuracy
    print(f"{'Task Name':<40} | {'Best Module':<18} | {'Peak Accuracy':<12}")
    print("-" * 75)

    for _, row in best_modules.iterrows():
        task = row["task_name"]
        module = row["module"]
        accuracy = row["accuracy_pct"]

        print(f"{task:<40} | {module:<18} | {accuracy:>6.2f}%")

    print("-" * 75)


def plot_vision_representation_profile(
    model_name, model_size="1B", patch_size=16, resolution=224, output_dir="figures"
):
    """Generates a grid of line plots (one per dataset) showing module accuracy (%)

    vs layer depth (%) for vision models.
    """
    path = RESULT_DIR / "image_classification.csv"  # Adjust file name as needed
    df = pd.read_csv(path)

    # 1. Filter by model architecture metadata
    filtered_df = df[
        (df["model_name"] == model_name)
        & (df["model_size"] == model_size)
        & (df["patch_size"] == patch_size)
        & (df["resolution"] == resolution)
    ].copy()

    if filtered_df.empty:
        print("Error: No data found for the specified vision model configuration.")
        return

    unique_datasets = sorted(filtered_df["dataset_name"].unique())
    num_datasets = len(unique_datasets)

    # 2. Process scales (Accuracy -> % and Layer -> Depth %)
    processed_dfs = []
    for dataset in unique_datasets:
        ds_df = filtered_df[filtered_df["dataset_name"] == dataset].copy()
        ds_df["accuracy_pct"] = ds_df["accuracy"] * 100

        max_layer = ds_df["layer"].max()
        ds_df["layer_pct"] = (ds_df["layer"] / max_layer) * 100 if max_layer > 0 else 0.0
        processed_dfs.append(ds_df)

    filtered_df = pd.concat(processed_dfs)

    # 3. Dynamic Grid setup (e.g., 2x5 grid for 10 datasets)
    ncols = 5
    nrows = math.ceil(num_datasets / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4 * nrows + 2))
    axes = axes.flatten()

    legend_dict = {}

    # 4. Plot curves per dataset
    for i, dataset in enumerate(unique_datasets):
        ax = axes[i]
        ds_data = filtered_df[filtered_df["dataset_name"] == dataset]

        # Aggregate across experimental runs if multiple exist per layer/module
        grouped = ds_data.groupby(["module", "layer_pct"])["accuracy_pct"].mean().reset_index()

        for module_name, mod_df in grouped.groupby("module"):
            mod_df = mod_df.sort_values("layer_pct")
            (line,) = ax.plot(
                mod_df["layer_pct"],
                mod_df["accuracy_pct"],
                label=module_name,
                linewidth=2,
                alpha=0.9,
            )
            if module_name not in legend_dict:
                legend_dict[module_name] = line

        ax.set_title(dataset, fontsize=12, weight="bold", pad=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Clean up empty subplots
    for j in range(num_datasets, len(axes)):
        fig.delaxes(axes[j])

    # Global labels and Legend
    fig.text(0.5, 0.02, "Layer Depth (%)", ha="center", fontsize=14, weight="semibold")
    fig.text(
        0.01,
        0.5,
        "Accuracy (%) [Mean]",
        va="center",
        rotation="vertical",
        fontsize=14,
        weight="semibold",
    )

    sorted_modules = sorted(legend_dict.keys())
    fig.legend(
        [legend_dict[m] for m in sorted_modules],
        sorted_modules,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(sorted_modules),
        fontsize=11,
    )

    title_str = f"Vision Profile: {model_name} ({model_size}, Patch {patch_size}, {resolution}px)"
    fig.suptitle(title_str, fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_name = f"vision_separated_{model_name}_{model_size}_p{patch_size}.png"
    plt.savefig(Path(output_dir) / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_vision_aggregated_profile(model_name, model_size="1B", patch_size=16, resolution=224, output_dir="figures"):
    """Generates a single line plot showing vision model performance averaged across

    all datasets, with one curve per module across layer depth (%).
    """
    path = RESULT_DIR / "image_classification.csv"
    df = pd.read_csv(path)

    filtered_df = df[
        (df["model_name"] == model_name)
        & (df["model_size"] == model_size)
        & (df["patch_size"] == patch_size)
        & (df["resolution"] == resolution)
    ].copy()

    if filtered_df.empty:
        print("Error: No data found for specified configuration.")
        return

    unique_datasets = filtered_df["dataset_name"].unique()
    processed_dfs = []

    # Map layer depth % individually per dataset
    for dataset in unique_datasets:
        ds_df = filtered_df[filtered_df["dataset_name"] == dataset].copy()
        ds_df["accuracy_pct"] = ds_df["accuracy"] * 100

        max_layer = ds_df["layer"].max()
        ds_df["layer_pct"] = (ds_df["layer"] / max_layer) * 100 if max_layer > 0 else 0.0

        ds_grouped = ds_df.groupby(["module", "layer_pct"])["accuracy_pct"].mean().reset_index()
        processed_dfs.append(ds_grouped)

    # Average across all dataset lines
    aggregated_df = pd.concat(processed_dfs).groupby(["module", "layer_pct"])["accuracy_pct"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    for module_name, mod_data in aggregated_df.groupby("module"):
        mod_data = mod_data.sort_values("layer_pct")
        ax.plot(
            mod_data["layer_pct"],
            mod_data["accuracy_pct"],
            label=module_name,
            linewidth=2.5,
            alpha=0.9,
        )

    ax.set_title(
        f"Aggregated Vision Representation Profile\nModel: {model_name} {model_size} (Averaged over {len(unique_datasets)} Datasets)",
        fontsize=13,
        weight="bold",
        pad=12,
    )
    ax.set_xlabel("Layer Depth (%)", fontsize=11, weight="semibold")
    ax.set_ylabel("Mean Accuracy (%)", fontsize=11, weight="semibold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Modules", loc="best")

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_name = f"vision_aggregated_{model_name}_{model_size}_p{patch_size}.png"
    plt.savefig(Path(output_dir) / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_vision_top_modules_per_dataset(model_name, model_size="1B", patch_size=16, resolution=224):
    """Computes and prints top module performance and structural metrics

    per dataset for vision models in a clean text-table format.
    """
    path = RESULT_DIR / "image_classification.csv"  # Ensure RESULT_DIR is defined globally
    df = pd.read_csv(path)

    # 1. Filter for the specific vision model configuration
    model_df = df[
        (df["model_name"] == model_name)
        & (df["model_size"] == model_size)
        & (df["patch_size"] == patch_size)
        & (df["resolution"] == resolution)
    ].copy()

    if model_df.empty:
        print(f"Error: No data found for vision configuration: {model_name}")
        return

    # 2. Convert accuracy to percentage (0-100)
    model_df["accuracy_pct"] = model_df["accuracy"] * 100

    # 3. Step A: Average across the 10 experimental runs per group
    run_averages = model_df.groupby(["dataset_name", "module", "layer"])["accuracy_pct"].mean().reset_index()

    unique_datasets = sorted(run_averages["dataset_name"].unique())

    # 4. Print Table Headers
    print(f"\nTop Performing Modules per Dataset for Model: {model_name} ({model_size})")
    print("-" * 115)
    print(
        f"{'Dataset Name':<25} | {'Best Module':<12} | {'Rank (Peak)':<15} | {'Peak Layer':<12} | {'Last Layer Perf':<18} | {'Overall Rank'}"
    )
    print("-" * 115)

    # 5. Process each dataset row-by-row
    for dataset in unique_datasets:
        ds_df = run_averages[run_averages["dataset_name"] == dataset]
        max_layer = ds_df["layer"].max()

        peaks = {}
        peak_layers = {}
        last_layer_perfs = {}

        # Loop through modules present in this dataset to harvest performance peaks
        for mod in ds_df["module"].unique():
            mod_df = ds_df[ds_df["module"] == mod]

            # Peak performance across all layers
            idx_max = mod_df["accuracy_pct"].idxmax()
            peaks[mod] = mod_df.loc[idx_max, "accuracy_pct"]
            peak_layers[mod] = mod_df.loc[idx_max, "layer"]

            # Performance at the final output layer
            last_perf = mod_df[mod_df["layer"] == max_layer]["accuracy_pct"]
            last_layer_perfs[mod] = last_perf.values[0] if not last_perf.empty else 0.0

        # Rank the modules relative to each other based on peak accuracy
        sorted_modules = sorted(peaks.items(), key=lambda x: x[1], reverse=True)
        mod_ranks = {mod: i + 1 for i, (mod, _) in enumerate(sorted_modules)}

        # The absolute winner is the top element in our sorted list
        best_mod, best_peak = sorted_modules[0]
        best_layer = peak_layers[best_mod]
        best_last = last_layer_perfs[best_mod]

        # Construct the overall rank summary block for the final column
        # Example tracking style: act(#1) > fc2(#2) > fc1(#3)
        rank_summary = " > ".join([f"{mod}(#{rank})" for mod, rank in mod_ranks.items()])

        # Print the row cleanly using structured string spacing formatting
        print(
            f"{dataset:<25} | "
            f"{best_mod:<12} | "
            f"#{mod_ranks[best_mod]} ({best_peak:>5.2f}%) | "
            f"L{int(best_layer):<10} | "
            f"{best_last:>6.2f}%{'':<11} | "
            f"{rank_summary}"
        )

    print("-" * 115)


# model = "Qwen3-1.7B"  # "gemma-3-4b-pt"  # "Llama-3.2-3B"
# plot_global_representation_profile(model=model)
# plot_aggregated_representation_profile(model=model)
# print_top_modules_per_task(model=model)

model_name = "dinov3"  # "aimv2"
model_size = "vit7b16"  # "1B"
plot_vision_representation_profile(model_name=model_name, model_size=model_size)
plot_vision_aggregated_profile(model_name=model_name, model_size=model_size)
print_vision_top_modules_per_dataset(model_name=model_name, model_size=model_size)
