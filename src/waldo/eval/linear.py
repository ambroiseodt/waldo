r"""
Linear probing to evaluate VFMs and VLMs.

Notes
-----
We follow DINOv2 [1]_ exprimental setup to define the default parameters of logistic regression solver.

References
----------
.. [1] M. Oquab et al. DINOv2: Learning Robust Visual Feature without Supervision. In TMLR 2024

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from cuml.linear_model import LogisticRegression as cuMLLogisticRegression
from cuml.preprocessing import StandardScaler as cuMLStandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sympy import re
from torch.utils.data import DataLoader

from waldo.infra import DEVICE, SAVING_DIR

logger = logging.getLogger(__name__)

# Paths
TEMP_DIR = SAVING_DIR / "temp"


# ------------------------------------------------------------------------------
#   Config
# ------------------------------------------------------------------------------
@dataclass
class LinearProbingConfig:
    r"""
    Linear probing configuration file with Logistic Regression.

    Parameters
    ----------
    tol: float, default=1e-4
       Tolerance for stopping criteria.
    max_iter: int, default=100
        Maximum number of iterations to converge.
    device: str, default=DEVICE
        Device to use.
    seed: int, default=42
        Random seed for reproducibility.

    See also
    --------
    Official documentation of sklearn LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    Official documentation of cuml LogisticRegression: https://docs.rapids.ai/api/cuml/nightly/api/generated/cuml.linear_model.logisticregression/
    """

    # Classifier
    tol: float = 1e-4
    max_iter: int = 100

    # Temporary path
    temp_path: Path = TEMP_DIR / "temp"

    # Device
    device: str = DEVICE

    # Reproducibility
    seed: int = 42

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Valid tolerance
        assert self.tol >= 0, f"Invalid tolerance {self.tol}. This should be a nonnegative number."

        # Create temporary path
        self.temp_dir = TEMP_DIR / "embeddings"
        self.temp_dir.mkdir(exist_ok=True, parents=True)


# ------------------------------------------------------------------------------
#   Linear probing
# ------------------------------------------------------------------------------
class LinearProbing:
    r"""
    Linear probing with a logistic regression using the LBFGS solver.

    Default parameters follow DINOv2 [1]_, with gpu-friendly implementation with cuML available when GPU is available.
    Although StandardScaler is not used in [1]_, we follow common practice and normalize features before classification.

    Parameters
    ----------
    config: configuration class with
        tol: float, default=1e-4
        Tolerance for stopping criteria.
        max_iter: int, default=100
            Maximum number of iterations to converge.
        device: str, default=DEVICE
            Device to use.
        seed: int, default=42
            Random seed for reproducibility.

    References
    ----------
    .. [1] M. Oquab et al. DINOv2: Learning Robust Visual Feature without Supervision. In TMLR 2024
    """

    def __init__(self, config: LinearProbingConfig):
        super().__init__()
        self.tol = config.tol
        self.max_iter = config.max_iter
        self.device = config.device
        self.seed = config.seed
        self.temp_dir = config.temp_dir

    @torch.inference_mode()
    def _extract_embeddings(self, model: nn.Module, loader: DataLoader, split: str) -> None:
        model.eval()
        exposed_keys = []
        label_buffer = []

        logger.info(f"Extract embeddings for split: {split}")

        # Loop over batches
        for idx, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = map(lambda x: x.to(self.device), (x_batch, y_batch))

            # Recover features
            embeddings = model.get_embeddings(x_batch)

            for key, embedding in embeddings.item():
                # Recover layer index and module
                match = re.match(r"block_(\d+)_(.+)", key)
                layer, module = int(match.group(1)), str(match.group(2))

                # Move embedding to numpy (output of get_embeddings are expected to be on cpu)
                embedding = embeddings[key].numpy()

                # Save batch of embedding to disk
                data_path = self.temp_dir / f"{split}" / f"layer_{layer}" / f"module_{module}" / f"batch_{idx}.npy"
                np.save(data_path, embedding)

            # Recover labels
            label_buffer.append(y_batch.cpu())

            # Clean memory
            del x_batch, y_batch, embeddings
            gc.collect()
            if "cuda" in self.device and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save labels to disk
        labels = torch.cat(label_buffer, dim=0).numpy()
        label_path = self.temp_dir / f"{split}" / "labels.npy"
        np.save(label_path, labels)

        # Clear memory
        del labels
        gc.collect()

        logger.info(f"Extraction done. {len(exposed_keys)} keys were exposed.")

    def _pipeline(self) -> Pipeline:
        # GPU-friendly logistic regression
        if torch.cuda.is_available():
            scaler = cuMLStandardScaler()
            solver = cuMLLogisticRegression(tol=self.tol, max_iter=self.max_iter)

        # Sklearn logistic regression
        else:
            scaler = StandardScaler()
            solver = LogisticRegression(tol=self.tol, max_iter=self.max_iter, random_state=self.seed)

        return make_pipeline(scaler, solver)

    def _run(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
    ) -> float:
        clf = self._pipeline()
        clf.fit(train_embeddings, train_labels.ravel())
        acc_test = clf.score(test_embeddings, test_labels.ravel())
        return acc_test

    def _save(
        self, metrics: dict[int, dict[str, float]], config: LinearProbingConfig, dataset_name: str, model_name: str
    ) -> None:
        r"""
        Directly saves results and the configuration file to the simple flat layout.

        Saves paths matching:
        savings/runs/{dataset_name}/{model_name}/config.json
        savings/runs/{dataset_name}/{model_name}/layer_{id}_module_{name}/eval.json
        """
        from dataclasses import asdict

        from waldo.infra.parser import json_serializable

        # 1. Target the absolute directory root: e.g., savings/runs/cifar10/google/dinov3-7b
        model_dir = SAVING_DIR / "runs" / dataset_name / model_name
        model_dir.mkdir(exist_ok=True, parents=True)

        # 2. Restore the missing config.json file right at the model root folder level
        with open(model_dir / "config.json", "w") as f:
            json.dump(json_serializable(asdict(config)), f, indent=4)

        # 3. Iterate cleanly over our nested metrics map for the individual modules
        for layer_idx, modules in metrics.items():
            for module_name, score in modules.items():
                # Construct directory path matching your exact spec: layer_1_module_mha
                subfolder_name = f"layer_{layer_idx}_module_{module_name}"
                target_dir = model_dir / subfolder_name
                target_dir.mkdir(exist_ok=True, parents=True)

                # Write out the target evaluation payload dictionary
                eval_payload = {"test_accuracy": round(score, 5)}

                with open(target_dir / "eval.json", "w") as f:
                    json.dump(eval_payload, f, indent=4)

        logger.info(f"Configuration and evaluation payloads safely checkpointed under: {model_dir}")

    def evaluate(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, cache_path: str) -> None:
        # Extract train and test embeddings
        self._extract_embeddings(model=model, loader=train_loader, split="train")
        self._extract_embeddings(model=model, loader=test_loader, split="test")

        # Run linear probing over each layer and module
        train_path = self.temp_dir / "train"
        test_path = self.temp_dir / "test"
        regex = re.compile(r"batch_(\d+)")

        # Recover labels
        train_labels = np.load(train_path / "labels.npy").ravel()
        test_labels = np.load(test_path / "labels.npy").ravel()

        # Loop over layers
        layer_folders = [f.name for f in (train_path).iterdir() if f.is_dir()]
        for layer_folder in layer_folders:
            layer = int(layer_folder.split("_")[1])

            # Loop over modules
            module_folders = [f.name for f in (train_path / layer_folder).iterdir() if f.is_dir()]
            for module_folder in module_folders:
                module = module_folder.split("_")[1]

                # Train and test folder names
                train_path = train_path / layer_folder / module_folder
                test_path = test_path / layer_folder / module_folder

                # Sort batch files in numerical order to remain consistent with labels
                sort_key = lambda p: int(p.stem.split("_")[1])
                train_files = sorted(train_path.glob("batch_*.npy"), key=sort_key)
                test_files = sorted(test_path.glob("batch_*.npy"), key=sort_key)

                # Recover train and test embeddings
                train_embeddings = np.concatenate([np.load(f) for f in train_files], axis=0)
                test_embeddings = np.concatenate([np.load(f) for f in test_files], axis=0)

                # Run linear probing
                logging.info(f"Linear probing on layer {layer} and module {module}.")
                metrics = self._run(
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                    test_embeddings=test_embeddings,
                    test_labels=test_labels,
                )

        logger.info("Linear probing done.")

        # Save results
        self._save(metrics=metrics, cache_path=cache_path)
