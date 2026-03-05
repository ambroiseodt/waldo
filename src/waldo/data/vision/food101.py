r"""
Food101 Dataset.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import dataclass
from pathlib import Path

import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from ...config import DATASET_DIR


@dataclass
class Food101DatasetConfig:
    r"""
    Food101 configuration file.

    Parameters
    ----------
    save_dir: str
        Path from where to load the data or save them if it does not exit.
    mode: str
        Mode. Options are "train" and "test".
    transform: Any
        Transformation to apply to the images.
    """

    save_dir: Path = DATASET_DIR / "food101"
    mode: str = "train"
    transform: Compose | None = None

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        assert self.mode in ["train", "test"], f"Invalid mode {self.mode}. Options are 'train' and 'test."


class Food101Dataset(Dataset):
    r"""
    Food101 dataset from [1]_.

    It consits of 101_000 images of food with 101 classes, with 100 images per class,
    with 75_750 training images and 25_250 test images.

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
    FGVCAircraft was created by Lukas Bossard, Matthieu Guillaumin, Luc Van Gool.
    Official website: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/.

    References
    ----------
    .. [1] L. Bossard et al. Food-101 -- Mining Discriminative Components with Random Forests. In ECCV 2014
    """

    def __init__(self, config: Food101DatasetConfig):
        super().__init__()
        split = config.mode
        dataset = torchvision.datasets.Food101(
            root=str(config.save_dir),
            split=split,
            download=True,
        )

        # Recover images and corresponding labels
        self.data = dataset._image_files
        self.targets = dataset._labels
        self.loader = dataset.loader
        self.n_classes = 100

        # Recover transform
        self.transform = config.transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.loader(self.data[idx])
        label = self.targets[idx]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __repr__(self):
        return f"Dataset with {len(self.data)} images."
