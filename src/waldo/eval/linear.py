r"""
Linear probing to evaluate VFMs and VLMs on image classification tasks.

Notes
-----
We follow DINOv2 [1]_ exprimental setup to define the default parameters of the logistic regression.

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
import shutil
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

from ..infra import DEVICE
from ..models import BaseVisionLanguageModel, BaseVisionModel

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
#   Config
# ------------------------------------------------------------------------------
@dataclass
class LinearProbingConfig:
    r"""
    Linear probing configuration file with Logistic Regression.
    The default logistic regression parameters follow DINOv2 [1]_.

    Parameters
    ----------
    tol: float, default=1e-12
       Tolerance for stopping criteria.
    max_iter: int, default=1000
        Maximum number of iterations to converge.
    device: str, default=DEVICE
        Device to use.
    seed: int, default=42
        Random seed for reproducibility.

    References
    ----------
    .. [1] M. Oquab et al. DINOv2: Learning Robust Visual Feature without Supervision. In TMLR 2024

    See also
    --------
    Official documentation of sklearn LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    Official documentation of cuml LogisticRegression: https://docs.rapids.ai/api/cuml/nightly/api/generated/cuml.linear_model.logisticregression/
    """

    # Classifier
    tol: float = 1e-12
    max_iter: int = 1000

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
        tol: float, default=1e-12
        Tolerance for stopping criteria.
        max_iter: int, default=1000
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

    @staticmethod
    def _get_index(path: Path) -> int:
        r"""Extract the index from a path by splitting the stem string."""
        # This was the line causing the IndexError when used on folder names like "layer_x"
        return int(path.stem.split("batch_")[1])

    @torch.inference_mode()
    def _extract_embeddings(
        self, model: BaseVisionModel | BaseVisionLanguageModel, loader: DataLoader, split: str, temp_path: Path
    ) -> None:
        r"""
        Extract embeddings of a model accross its layers and modules.
        The model is expected to be on the device defined in the config.

        Parameters
        ----------
        model: BaseVisionModel | BaseVisionLanguageModel
            Model to evaluate.
        loader: DataLoader
            DataLoader to use for evaluation.
        split: str
            Split to evaluate on. Options are "train" and "test".
            This is used to save the embeddings temporarily on the disk.

        See also
        --------
        BaseVisionModel
        BaseVisionLanguageModel
        """
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
                embedding = embedding.numpy()

                # Save batch of embedding to disk
                data_path = temp_path / f"{split}" / f"layer_{layer}" / f"module_{module}"
                data_path.mkdir(parents=True, exist_ok=True)
                np.save(data_path / f"batch_{idx}.npy", embedding)

            # Recover labels
            label_buffer.append(y_batch.cpu())

            # Clean memory
            del x_batch, y_batch, embeddings
            gc.collect()
            if "cuda" in self.device and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save labels to disk
        labels = torch.cat(label_buffer, dim=0).numpy()
        label_path = temp_path / f"{split}" / "labels.npy"
        np.save(label_path, labels)

        # Clear memory
        del labels
        gc.collect()

        logger.info("Extraction done.")

    def _pipeline(self) -> Pipeline:
        r"""
        Return a sklearn Pipeline with StandardScaler and LogisticRegression.
        If GPU is available, use cuML implementation for faster training.
        """
        # GPU-friendly logistic regression
        if torch.cuda.is_available():
            scaler = cuMLStandardScaler()
            solver = cuMLLogisticRegression(
                penalty="l2",
                tol=self.tol,
                C=1.0,
                max_iter=self.max_iter,
                linesearch_max_iter=50,
            )

        # Sklearn logistic regression
        else:
            scaler = StandardScaler()
            solver = LogisticRegression(
                penalty="l2",
                tol=self.tol,
                C=1.0,
                max_iter=self.max_iter,
                random_state=self.seed,
            )

        return make_pipeline(scaler, solver)

    def _run(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
    ) -> float:
        r"""Run linear probing with logistic regression and return the test accuracy.

        Parameters
        ----------
        train_embeddings: np.ndarray
            Train embeddings of shape (n_samples, n_features).
        train_labels: np.ndarray
            Train labels of shape (n_samples,).
        test_embeddings: np.ndarray
            Test embeddings of shape (n_samples, n_features).
        test_labels: np.ndarray
            Test labels of shape (n_samples,).

        Returns
        -------
        float:
            Test accuracy of the linear probe.
        """

        clf = self._pipeline()
        clf.fit(train_embeddings, train_labels.ravel())
        acc_test = clf.score(test_embeddings, test_labels.ravel())
        return float(acc_test)

    def evaluate(
        self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, cache_path: Path, overwrite: bool
    ) -> None:
        r"""
        Evaluate the model with linear probing on the given train and test dataloaders.
        Embeddings are saved temporarily on disk to reduce memory usage and delete after evaluation.
        Results are saved to the specified cache path.
        """
        # Overwrite existing results
        if overwrite:
            if cache_path.exists():
                logger.info(f"Overwriting enabled. Deleting existing cache folder at: '{cache_path}'")
                shutil.rmtree(cache_path)
                cache_path.mkdir(parents=True, exist_ok=True)

        # Do not overwrite existing results and skip evaluation loop
        else:
            if cache_path.exists():
                logger.info(
                    f"The entire evaluation loop is skipped: found existing results folder at: '{cache_path}'."
                    f"Please, verify the completeness and viability of these files."
                    f"If this behavior is not expected, set overwrite=True to force a clean rerun."
                )
                return

        # Create temporary path
        temp_path = cache_path / "temp_embeddings"
        temp_path.mkdir(exist_ok=True, parents=True)

        try:
            # Extract train and test embeddings
            self._extract_embeddings(model=model, loader=train_loader, split="train", temp_path=temp_path)
            self._extract_embeddings(model=model, loader=test_loader, split="test", temp_path=temp_path)

            # Run linear probing over each layer and module
            train_path = temp_path / "train"
            test_path = temp_path / "test"

            # Recover labels
            train_labels = np.load(train_path / "labels.npy").ravel()
            test_labels = np.load(test_path / "labels.npy").ravel()

            # Loop over layers
            layer_folders = [f.name for f in train_path.iterdir() if f.is_dir()]
            print("folders", layer_folders)
            for layer_folder in layer_folders:
                layer = int((layer_folder).split("_")[1])

                # Loop over modules
                module_folders = [f.name for f in (train_path / layer_folder).iterdir() if f.is_dir()]
                for module_folder in module_folders:
                    print("module", module_folder)
                    module = module_folder.split("_")[1]

                    # Train and test folder names
                    train_emb_path = train_path / layer_folder / module_folder
                    test_emb_path = test_path / layer_folder / module_folder

                    # Sort batch files in numerical order to remain consistent with labels
                    train_files = sorted(train_emb_path.glob("batch_*.npy"), key=self._get_index)
                    test_files = sorted(test_emb_path.glob("batch_*.npy"), key=self._get_index)
                    print("batch", [f.name for f in train_files])
                    # Recover train and test embeddings
                    train_embeddings = np.concatenate([np.load(f) for f in train_files], axis=0)
                    test_embeddings = np.concatenate([np.load(f) for f in test_files], axis=0)

                    # Run linear probing
                    logging.info(f"Linear probing of layer {layer} and module {module}.")
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

                    # Delete temporary files
                    shutil.rmtree(train_emb_path, ignore_errors=True)
                    shutil.rmtree(test_emb_path, ignore_errors=True)

                    # Clean memory
                    del train_embeddings, test_embeddings
                    gc.collect()
                    if "cuda" in self.device and torch.cuda.is_available():
                        torch.cuda.empty_cache()

        finally:
            # Delete temporary files
            if temp_path.exists():
                shutil.rmtree(temp_path)
                logger.info("Delete temporary files.")
