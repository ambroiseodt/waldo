r"""
EuroSAT Dataset.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from ...config import DATASET_DIR
from .utils import deterministic_split


@dataclass
class EuroSATDatasetConfig:
    r"""
    EuroSAT configuration file.

    Parameters
    ----------
    save_dir: str
        Path from where to load the data or save them if it does not exit.
    mode: str
        Mode. Options are "train" and "test".
    transform: Any
        Transformation to apply to the images.
    """

    save_dir: Path = DATASET_DIR / "eurosat"
    mode: str = "train"
    transform: Compose | None = None

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.mode not in ["train", "test"]:
            raise ValueError(f"Invalid mode '{self.mode}'. Options are 'train' and 'test'.")


class EuroSATDataset(Dataset):
    r"""
    EuroSAT dataset from [1]_.

    It consists of 21_600 RGB Sentinel-2 satellite images with 10 classes.
    Since there is no official train or test sets, we create a deterministic 80/20 split.

    Parameters
    ----------
    config: configuration class with
        save_dir: str
            Path from where to load the data or save them if it does not exit.
        mode: str
            Mode. Options are "train" and "test".
        transform: Compose
            Transformation to apply to the images.

    Notes
    ----------
    EuroSAT was created by Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth.
    Official website: https://github.com/phelber/EuroSAT.

    Download
    --------
    The dataset is downloaded automatically via torchvision when download=True.

    References
    ----------
    .. [1] P. Helber et al. EuroSAT: A novel dataset and deep learning benchmark for land use
           and land cover classification. In IEEE Journal of Selected Topics in Applied Earth
           Observations and Remote Sensing, 2019
    """

    def __init__(self, config: EuroSATDatasetConfig):
        super().__init__()
        train = True if config.mode == "train" else False

        # Recover dataset
        dataset = torchvision.datasets.EuroSAT(
            root=str(config.save_dir),
            download=True,
        )
        samples = np.asarray(dataset.imgs)

        # Recover the (deterministic) train or test set
        indices = deterministic_split(data=samples, train_size=0.8, train=train)
        self.samples = samples[indices]
        self.n_classes = 10

        # Recover transform
        self.transform = config.transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        sample = Image.open(path).convert("RGB")
        label = int(label)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __repr__(self):
        return f"Dataset with {len(self.samples)} images."
