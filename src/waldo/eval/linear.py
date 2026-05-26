r"""
Codebase to perform linear probing on large vision transformers (e.g., DINOv3 7B)
optimized for shared high-performance computing clusters.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import gc
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from core.config import DEVICE, SAVING_DIR
from core.utils import json_serializable
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)
PROBE_DIR = SAVING_DIR / "probes"


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
@dataclass
class LinearProbingConfig:
    r"""Linear probing configuration payload."""

    # Experiment Info
    model_name: str = "dinov3"
    model_size: str = "7b"
    log_dir: str = ""
    checkpoint_step: str | None = None
    finetuned: bool = True
    cls_pooling: bool = False

    # Data
    dataset_name: str = "cifar10"
    train_size: float = 0.8
    batch_size: int = 512
    val_batch_size: int = 512
    test_batch_size: int = 512

    # Shared Cluster Resource Budgeting
    # Set these defensively so parallel jobs on the same machine don't collide
    num_workers: int = 4  # Number of CPU workers per dataloader
    cpu_cores_fit: int = 8  # Number of CPU threads for scikit-learn fitting

    # Environment
    device: str = DEVICE
    seed: int = 42
    logging_level: str = "INFO"

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        self.log_dir = Path(self.log_dir) if self.log_dir else Path("")
        if self.seed is None:
            self.seed = 42


# ------------------------------------------------------------------------------
# Feature Extraction (VRAM Optimized)
# ------------------------------------------------------------------------------
@torch.inference_mode()
def extract_split_embeddings(
    model: nn.Module, loader: DataLoader, cls_pooling: bool, device: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    r"""
    Extracts internal hidden state probes batch-by-batch.
    Down-samples spatially on the GPU accelerator immediately to protect RAM.
    """
    model.eval()
    grouped_embeddings: dict[str, list[np.ndarray]] = {}
    label_list: list[np.ndarray] = []

    for x_batch, y_batch in tqdm(loader, desc="Extracting features", leave=False):
        # Cast inputs cleanly to FP16 to match large model weights
        x_batch = x_batch.to(device=device, dtype=torch.float16)

        # Opaque probe extraction pass
        probes = model.get_probes(x_batch)

        for key, tensor_block in probes.items():
            # Spatial dimension pooling execution on hardware accelerator
            if cls_pooling:
                pooled_emb = tensor_block[:, 0, :]
            else:
                pooled_emb = tensor_block.mean(dim=1)

            # Detach cleanly, copy to host system RAM, cast to float32
            cpu_numpy = pooled_emb.detach().cpu().to(torch.float32).numpy()

            if key not in grouped_embeddings:
                grouped_embeddings[key] = []
            grouped_embeddings[key].append(cpu_numpy)

        label_list.append(y_batch.detach().cpu().numpy())

    # Consolidate and unit-normalize embeddings
    final_embeddings: dict[str, np.ndarray] = {}
    for key, blocks in grouped_embeddings.items():
        concatenated = np.concatenate(blocks, axis=0)
        concatenated /= np.linalg.norm(concatenated, axis=-1, keepdims=True) + 1e-8
        final_embeddings[key] = concatenated

    final_labels = np.concatenate(label_list, axis=0)
    return final_embeddings, final_labels


def run_linear_probing(
    model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, config: LinearProbingConfig
) -> dict[str, float]:
    r"""Runs full evaluation and handles strict memory hand-offs."""

    # Step 1: Feature extraction across splits
    logger.info("Extracting training representations...")
    train_embeddings, train_labels = extract_split_embeddings(
        model=model, loader=train_loader, cls_pooling=config.cls_pooling, device=config.device
    )

    logger.info("Extracting validation/test representations...")
    test_embeddings, test_labels = extract_split_embeddings(
        model=model, loader=test_loader, cls_pooling=config.cls_pooling, device=config.device
    )

    # ---------------------------------------------------------------------
    # HARD RESET: Evict massive model & data loading workers from the host
    # and device before allocating memory for downstream solvers.
    # ---------------------------------------------------------------------
    logger.info("Purging foundation model and dataloader allocations from hardware caches...")
    del model
    del train_loader, test_loader
    gc.collect()
    if "cuda" in str(config.device) and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 2: Fit linear classifiers on CPU using bounded thread constraints
    metrics: dict[str, float] = {}
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=config.seed,
            n_jobs=config.cpu_cores_fit,  # Strict cap to prevent thrashing shared nodes
            solver="lbfgs",
        ),
    )

    for key in tqdm(train_embeddings.keys(), desc="Training layer probes"):
        X_train = train_embeddings[key]
        X_test = test_embeddings[key]

        clf.fit(X_train, train_labels.ravel())
        accuracy = clf.score(X_test, test_labels.ravel())
        metrics[key] = float(accuracy)

    return metrics


# ------------------------------------------------------------------------------
# MTEB-Styled High-Fidelity Results Archiver
# ------------------------------------------------------------------------------
def save_mteb_results(
    metrics: dict[str, float], config: LinearProbingConfig, elapsed_time: float, save_root: Path = PROBE_DIR
) -> Path:
    r"""Packages results into an invariant nested folder tree and metadata JSON schema."""
    model_folder = f"{config.model_name}_{config.model_size}"
    output_dir = save_root / config.dataset_name / model_folder / f"seed_{config.seed}"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Write reproducibility config
    with open(output_dir / "config.json", "w") as f:
        json.dump(json_serializable(asdict(config)), f, indent=4)

    # Convert dynamic internal strings to clean sub-attribute dictionaries
    scores_manifest = []
    for composite_key, accuracy in metrics.items():
        tokens = composite_key.split("_")
        layer_id = next((t for t in tokens if t.isdigit()), "0")
        module_name = "_".join([t for t in tokens if not t.isdigit()])

        scores_manifest.append(
            {
                "layer": int(layer_id),
                "module_component": module_name,
                "accuracy": round(accuracy, 5),
                "cls_pooling": bool(config.cls_pooling),
                "finetuned": bool(config.finetuned),
            }
        )

    # Centralized metadata tracking payload
    mteb_payload = {
        "mteb_style_version": "1.14.0",
        "generated_timestamp": datetime.utcnow().isoformat(),
        "dataset_name": config.dataset_name,
        "model_meta": {"model_name": config.model_name, "model_size": config.model_size},
        "environment": {
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "cuda_device": str(config.device),
        },
        "execution_stats": {
            "total_eval_time_seconds": round(elapsed_time, 2),
            "batch_size": config.batch_size,
            "dataloader_workers": config.num_workers,
            "sklearn_cores_used": config.cpu_cores_fit,
        },
        "scores": {"test": scores_manifest},
    }

    output_file = output_dir / "linear_probing.json"
    with open(output_file, "w") as f:
        json.dump(mteb_payload, f, indent=4)

    logger.info(f"Standardized MTEB report written cleanly to: {output_file}")
    return output_file
