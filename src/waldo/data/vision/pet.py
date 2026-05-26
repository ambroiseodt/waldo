r"""
Oxford-IIIT Pet Dataset.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import dataclass
from pathlib import Path

import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from ...infra import DATASET_DIR


# ------------------------------------------------------------------------------
#   Config
# ------------------------------------------------------------------------------
@dataclass
class PetDatasetConfig:
    r"""
    Oxford-IIIT Pet configuration file.

    Parameters
    ----------
    save_dir: str
        Path from where to load the data or save them if it does not exit.
    mode: str
        Mode. Options are "train" and "test".
    transform: Any
        Transformation to apply to the images.
    """

    save_dir: Path = DATASET_DIR / "pet"
    mode: str = "train"
    transform: Compose | None = None

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.mode not in ["train", "test"]:
            raise ValueError(f"Invalid mode '{self.mode}'. Options are 'train' and 'test'.")


# ------------------------------------------------------------------------------
#   Dataset
# ------------------------------------------------------------------------------
class PetDataset(Dataset):
    r"""
    Oxford-IIIT Pet dataset from [1]_.

    It consists of 37 category of pets with around 200 images for each class.
    The images have a large variations in scale, pose and lighting.

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
    Oxford-IIIT Pet was created by Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, C. V. Jawahar
    Official website: https://www.robots.ox.ac.uk/~vgg/data/pets/.

    References
    ----------
    .. [1] O. M. Parkhi et al. Cats and Dogs. In CVPR 2012
    """

    def __init__(self, config: PetDatasetConfig):
        super().__init__()
        split = "trainval" if config.mode == "train" else "test"
        dataset = torchvision.datasets.OxfordIIITPet(
            root=str(config.save_dir),
            split=split,
            download=True,
        )

        # Recover dataset
        self.data = dataset._images
        self.targets = dataset._labels
        self.n_classes = 37

        # Recover transform
        self.transform = config.transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        path = self.data[idx]
        sample = Image.open(path).convert("RGB")
        label = self.targets[idx]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __repr__(self):
        return f"Dataset with {len(self.data)} images."
