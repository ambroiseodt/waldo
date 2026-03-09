r"""
FGVCAircraft Dataset.

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
class AircraftDatasetConfig:
    r"""
    FGVCAircraft configuration file.

    Parameters
    ----------
    save_dir: str
        Path from where to load the data or save them if it does not exit.
    mode: str
        Mode. Options are "train" and "test".
    transform: Any
        Transformation to apply to the images.
    """

    save_dir: Path = DATASET_DIR / "aircraft"
    mode: str = "train"
    transform: Compose | None = None

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.mode not in ["train", "test"]:
            raise ValueError(f"Invalid mode '{self.mode}'. Options are 'train' and 'test'.")


class AircraftDataset(Dataset):
    r"""
    FGVCAircraft dataset from [1]_.

    It consits of 10_000 aircrafts images in 100 classes, with 100 images per class.

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
    FGVCAircraft was created by Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, Andrea Vedaldi.
    Official website: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/.

    References
    ----------
    .. [1] S. Maji et al. Fine-Grained Visual Classification of Aircraft. arXiv, 2013
    """

    def __init__(self, config: AircraftDatasetConfig):
        super().__init__()
        split = config.mode
        dataset = torchvision.datasets.FGVCAircraft(
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
