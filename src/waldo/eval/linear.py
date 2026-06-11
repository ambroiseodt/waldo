r"""
Linear probing to evaluate VFMs and VLMs on image classification tasks.

References
----------
.. [1] M. Oquab et al. DINOv2: Learning Robust Visual Feature without Supervision. In TMLR 2024

.. [2] O. Simeoni et al. DINOv3. arXiv 2025

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import gc
import json
import logging
import os
import re
import shutil
import time
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
from tqdm import tqdm

from ..infra import DEVICE
from ..models import BaseVisionLanguageModel, BaseVisionModel

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
#   Config
# ------------------------------------------------------------------------------
@dataclass
class LinearProbingConfig:
    r"""
    Linear probing configuration file with Logistic Regression following [1, 2]_.

    Parameters
    ----------
    tol: float, default=1e-4
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

    .. [2] O. Simeoni et al. DINOv3. arXiv 2025

    See also
    --------
    Official documentation of sklearn LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    Official documentation of cuml LogisticRegression: https://docs.rapids.ai/api/cuml/nightly/api/generated/cuml.linear_model.logisticregression/
    """

    # Classifier
    tol: float = 1e-4
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
    Linear probing with a logistic regression with GPU-friendly implementation.

    Parameters
    ----------
    config: configuration class with
        tol: float, default=1e-4
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

        logger.info(f"Extract {split} embeddings.")

        # Loop over batches
        for idx, (x_batch, y_batch) in enumerate(tqdm(loader, desc="Extraction", unit="batch")):
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
                output_type="numpy",
            )

        # Sklearn logistic regression
        else:
            scaler = StandardScaler()
            solver = LogisticRegression(
                C=1.0,
                l1_ratio=0.0,
                tol=self.tol,
                max_iter=self.max_iter,
                random_state=self.seed,
                solver="lbfgs",
            )
        return make_pipeline(scaler, solver)

    def _fit(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> Pipeline:
        r"""
        Train a linear classifier.

        Parameters
        ----------
        train_embeddings: np.ndarray
            Train embeddings of shape (n_samples, n_features).
        train_labels: np.ndarray
            Train labels of shape (n_samples,).

        Returns
        -------
        clf: Pipeline
             Fitted sklearn Pipeline.
        """

        clf = self._pipeline()
        clf.fit(train_embeddings, train_labels.ravel())
        return clf

    def _score(
        self,
        clf: Pipeline,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
    ) -> float:
        r"""
        Evaluate a linear classifier.

        Parameters
        ----------
        test_embeddings: np.ndarray
            Test embeddings of shape (n_samples, n_features).
        test_labels: np.ndarray
            Test labels of shape (n_samples,).

        Returns
        -------
        float:
            Test accuracy of the linear probe.
        """
        acc_test = clf.score(test_embeddings, test_labels.ravel())
        return float(acc_test)

    def _evaluate_single(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        cache_path: Path,
        overwrite: bool,
    ) -> None:
        r"""
        Evaluate the model with linear probing on a pair of train and test dataloaders.
        Embeddings are saved temporarily on disk to reduce memory usage and deleted
        after evaluation. Results are saved to the specified cache path.

        Parameters
        ----------
        model: nn.Module
            Model to evaluate. It is expected to be on the device defined in the config.
        train_loader: DataLoader
            Train dataloader.
        test_loader: DataLoader
            Test dataloader.
        cache_path: Path
            Path where the results are saved.
        overwrite: bool
            Whether to overwrite existing results.
        """
        # Check whether results already exists
        results_path = cache_path / "results"
        if results_path.exists():
            # Overwrite result folder
            if overwrite:
                logger.info(
                    f"Deleting cached results found at '{results_path}'. "
                    f"If this behavior is not expected, set overwrite=False to keep results and skip evaluation."
                )
                shutil.rmtree(results_path, ignore_errors=True)

            # Skip evaluation
            else:
                logger.info(
                    f"Skipping evaluation: results already exists at '{results_path}'. "
                    f"Please make sure these files are complete and viable. "
                    f"If this behavior is not expected, set overwrite=True to force a clean rerun."
                )
                return

        # Create temporary path
        temp_path = results_path / "temp_embeddings"

        try:
            # Extract train embeddings and recover label
            info = {}
            start = time.time()
            self._extract_embeddings(model=model, loader=train_loader, split="train", temp_path=temp_path)
            train_path = temp_path / "train"
            train_labels = np.load(train_path / "labels.npy").ravel()
            info["train_extraction_time"] = time.time() - start

            # Extract test embeddings and recover label
            start = time.time()
            self._extract_embeddings(model=model, loader=test_loader, split="test", temp_path=temp_path)
            test_path = temp_path / "test"
            test_labels = np.load(test_path / "labels.npy").ravel()
            info["test_extraction_time"] = time.time() - start

            # Loop over layers
            logger.info("Start evaluation.")
            start = time.time()
            total_runs = sum(1 for root, dirs, files in os.walk(train_path) if not dirs)
            with tqdm(total=total_runs, desc="Evaluation", unit="run") as pbar:
                layer_folders = [f.name for f in train_path.iterdir() if f.is_dir()]
                for layer_folder in layer_folders:
                    layer = int((layer_folder).split("_")[1])

                    # Loop over modules
                    module_folders = [f.name for f in (train_path / layer_folder).iterdir() if f.is_dir()]
                    for module_folder in module_folders:
                        module = module_folder.split("_")[1]

                        # Fit classifier on train embeddings
                        # Batch files are sorted numerically to remain consistent with labels
                        train_emb_path = train_path / layer_folder / module_folder
                        train_files = sorted(train_emb_path.glob("batch_*.npy"), key=self._get_index)
                        train_embeddings = np.concatenate([np.load(f) for f in train_files], axis=0)
                        clf = self._fit(
                            train_embeddings=train_embeddings,
                            train_labels=train_labels,
                        )

                        # Evaluate classifier on test embeddings
                        test_emb_path = test_path / layer_folder / module_folder
                        test_files = sorted(test_emb_path.glob("batch_*.npy"), key=self._get_index)
                        test_embeddings = np.concatenate([np.load(f) for f in test_files], axis=0)
                        acc_test = self._score(
                            clf=clf,
                            test_embeddings=test_embeddings,
                            test_labels=test_labels,
                        )

                        # Save result
                        metric_path = results_path / f"layer_{layer}_module_{module}.json"
                        # metric_path.parent.mkdir(exist_ok=True, parents=True)
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

                        # Update progress bar
                        pbar.update(1)

            # Save evaluation time
            info["eval_time"] = time.time() - start
            with open(cache_path / "info.json", "w") as f:
                json.dump(info, f, indent=4)
            logger.info("Evaluation done.")

        finally:
            # Delete temporary files
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
                logger.info("Delete temporary files.")

    def _evaluate_multiple(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: dict[str, DataLoader],
        cache_path: Path,
        overwrite: bool,
    ) -> None:
        r"""
        Evaluate the model with linear probing on a train dataloader and several test dataloaders.
        Evaluation is done independently on each test loader. Embeddings are saved temporarily on
        disk to reduce memory usage and deleted after evaluation.
        Results are saved to the specified cache path.

        Parameters
        ----------
        model: nn.Module
            Model to evaluate. It is expected to be on the device defined in the config.
        train_loader: DataLoader
            Train dataloader.
        test_loader: dict[str, DataLoader]
            Dictionary of test dataloaders where the keys specify which dataset they correspond to.
        cache_path: Path
            Path where the results are saved.
        overwrite: bool
            Whether to overwrite existing results.
        """
        # Check whether results already exists
        results_path = cache_path / "results"
        all_keys = set(test_loader.keys())
        for key in test_loader:
            path = results_path / key
            if path.exists():
                # Overwrite result folder
                if overwrite:
                    logger.info(
                        f"Deleting cached results found at '{path}'. "
                        f"If this behavior is not expected, set overwrite=False to keep results and skip evaluation."
                    )
                    shutil.rmtree(path, ignore_errors=True)
                # Skip evaluation
                else:
                    logger.info(
                        f"Skipping evaluation for {key}: results already exists at '{path}'. "
                        f"Please make sure these files are complete and viable. "
                        f"If this behavior is not expected, set overwrite=True to force a clean rerun."
                    )
                    all_keys.discard(key)

        # If overwrite is False andall results already exists, skip the evaluation
        if not all_keys:
            logger.info(
                "Skipping the entire evaluation loop: all results already exists and overwrite is set to False."
            )
            return

        # Create temporary path
        temp_path = results_path / "temp_embeddings"

        try:
            # Extract train embeddings and recover label
            info = {}
            start = time.time()
            self._extract_embeddings(model=model, loader=train_loader, split="train", temp_path=temp_path)
            train_path = temp_path / "train"
            train_labels = np.load(train_path / "labels.npy").ravel()
            info["train_extraction_time"] = time.time() - start

            # Extract test embeddings and recover label
            test_path = {}
            test_labels = {}
            start = time.time()
            for key in test_loader:
                split = f"test/{key}"
                self._extract_embeddings(model=model, loader=test_loader[key], split=split, temp_path=temp_path)
                test_path[key] = temp_path / split
                test_labels[key] = np.load(test_path[key] / "labels.npy").ravel()
            info["test_extraction_time"] = time.time() - start

            # Loop over layers
            logger.info("Start evaluation.")
            start = time.time()
            total_runs = sum(1 for root, dirs, files in os.walk(train_path) if not dirs)
            with tqdm(total=total_runs, desc="Evaluation", unit="run") as pbar:
                layer_folders = [f.name for f in train_path.iterdir() if f.is_dir()]
                for layer_folder in layer_folders:
                    layer = int((layer_folder).split("_")[1])

                    # Loop over modules
                    module_folders = [f.name for f in (train_path / layer_folder).iterdir() if f.is_dir()]
                    for module_folder in module_folders:
                        module = module_folder.split("_")[1]

                        # Fit classifier on train embeddings
                        # Batch files are sorted numerically to remain consistent with labels
                        train_emb_path = train_path / layer_folder / module_folder
                        train_files = sorted(train_emb_path.glob("batch_*.npy"), key=self._get_index)
                        train_embeddings = np.concatenate([np.load(f) for f in train_files], axis=0)
                        clf = self._fit(
                            train_embeddings=train_embeddings,
                            train_labels=train_labels,
                        )

                        # Evaluate classifier on test embeddings
                        for key in test_loader:
                            test_emb_path = test_path[key] / layer_folder / module_folder
                            test_files = sorted(test_emb_path.glob("batch_*.npy"), key=self._get_index)
                            test_embeddings = np.concatenate([np.load(f) for f in test_files], axis=0)
                            acc_test = self._score(
                                clf=clf,
                                test_embeddings=test_embeddings,
                                test_labels=test_labels[key],
                            )

                            # Save result
                            metric_path = results_path / key / f"layer_{layer}_module_{module}.json"
                            metric_path.parent.mkdir(exist_ok=True, parents=True)
                            with open(metric_path, "w") as f:
                                json.dump({"acc_test": acc_test}, f, indent=4)

                            # Delete temporary files
                            shutil.rmtree(train_emb_path, ignore_errors=True)
                            shutil.rmtree(test_emb_path, ignore_errors=True)

                            # Clean memory
                            del test_embeddings

                            # Update progress bar
                            pbar.update(1)

                        # Clean memory
                        del train_embeddings
                        gc.collect()
                        if "cuda" in self.device and torch.cuda.is_available():
                            torch.cuda.empty_cache()

            # Save evaluation time
            info["eval_time"] = time.time() - start
            with open(cache_path / "info.json", "w") as f:
                json.dump(info, f, indent=4)
            logger.info("Evaluation done.")

        finally:
            # Delete temporary files
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
                logger.info("Delete temporary files.")

    def evaluate(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader | dict[str, DataLoader],
        cache_path: Path,
        overwrite: bool,
    ) -> None:
        r"""
        Evaluate the model with linear probing on a pair of train and test dataloaders.
        Embeddings are saved temporarily on disk to reduce memory usage and deleted
        after evaluation. Results are saved to the specified cache path.

        Parameters
        ----------
        model: nn.Module
            Model to evaluate. It is expected to be on the device defined in the config.
        train_loader: DataLoader
            Train dataloader.
        test_loader: DataLoader | dict[str, DataLoader]
            Test dataloader(s). If test_loader is a dict, the evaluation is done on each
            dataloader independently and results are saved in
        cache_path: Path
            Path where the results are saved.
        overwrite: bool
            Whether to overwrite existing results.
        """
        # Evaluation on multiple test loaders (independently)
        if isinstance(test_loader, dict):
            self._evaluate_multiple(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                cache_path=cache_path,
                overwrite=overwrite,
            )
        # Evaluation on a single test loader
        else:
            self._evaluate_single(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                cache_path=cache_path,
                overwrite=overwrite,
            )
