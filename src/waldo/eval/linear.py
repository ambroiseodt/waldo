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
import json
import logging
import re
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
from torch.utils.data import DataLoader

from waldo.infra import DEVICE, SAVING_DIR
from waldo.models import BaseVisionModel

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

    @staticmethod
    def _get_index(path: Path) -> int:
        """Extracts the numerical index from a filename in format "batch_X"."""
        return int(path.stem.split("_")[1])

    @torch.inference_mode()
    def _extract_embeddings(self, model: BaseVisionModel, loader: DataLoader, split: str) -> None:
        model.eval()
        label_buffer = []

        logger.info(f"Extract embeddings for split: {split}")

        # Loop over batches
        for idx, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = map(lambda x: x.to(self.device), (x_batch, y_batch))

            # Recover features
            embeddings = model.get_embeddings(x_batch)

            for key, embedding in embeddings.items():
                # Recover layer index and module
                match = re.match(r"block_(\d+)_(.+)", key)
                if not match:
                    logger.warning(f"Skipping key '{key}' as it does not match the 'block_X_Y' format.")
                    continue
                else:
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

        logger.info("Extraction done.")

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
        return float(acc_test)

    def evaluate(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, cache_path: Path) -> None:
        # Extract train and test embeddings
        self._extract_embeddings(model=model, loader=train_loader, split="train")
        self._extract_embeddings(model=model, loader=test_loader, split="test")

        # Run linear probing over each layer and module
        train_path = self.temp_dir / "train"
        test_path = self.temp_dir / "test"

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
                train_emb_path = train_path / layer_folder / module_folder
                test_emb_path = test_path / layer_folder / module_folder

                # Sort batch files in numerical order to remain consistent with labels
                train_files = sorted(train_emb_path.glob("batch_*.npy"), key=self._get_index)
                test_files = sorted(test_emb_path.glob("batch_*.npy"), key=self._get_index)

                # Recover train and test embeddings
                train_embeddings = np.concatenate([np.load(f) for f in train_files], axis=0)
                test_embeddings = np.concatenate([np.load(f) for f in test_files], axis=0)

                # Run linear probing
                logging.info(f"Linear probing on layer {layer} and module {module}.")
                acc_test = self._run(
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                    test_embeddings=test_embeddings,
                    test_labels=test_labels,
                )

                # Save result
                metric_path = cache_path / f"layer_{layer}_module_{module}.json"
                metric_path.parent.mkdir(exist_ok=True, parents=True)
                with open(metric_path, "w") as f:
                    json.dump({"acc_test": acc_test}, f, indent=4)

        logger.info("Linear probing done.")
