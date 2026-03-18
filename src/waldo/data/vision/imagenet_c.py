r"""
ImageNet-C Dataset.

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
class ImageNetCDatasetConfig:
    r"""
    ImageNet-C configuration file.

    Parameters
    ----------
    save_dir: str
        Path from where to load the data.
    corruption_type: str, default="brightness"
        Type of corruption applied on the data. Options are:
            "brightness",
            "contrast",
            "defocus_blur",
            "elastic_transform",
            "fog",
            "frost",
            "gaussian_blur",
            "gaussian_noise",
            "glass_blur",
            "impulse_noise",
            "jpeg_compression",
            "motion_blur",
            "pixelate",
            "saturate",
            "shot_noise",
            "snow",
            "spatter",
            "speckle_noise",
            "zoom_blur".
    corruption_severity: int, default=1
        Severity of the corruption applied on the data.
        Allowed values are 1, 2, 3, 4 and 5.
    mode: str
        Mode. Options are "train" and "test".
    transform: Any
        Transformation to apply to the images.
    """

    save_dir: Path = DATASET_DIR / "imagenet_c"
    corruption_type: str = "brightness"
    corruption_severity: int = 1
    mode: str = "train"
    transform: Compose | None = None

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Valid corruption type
        if self.corruption_type not in [
            "brightness",
            "contrast",
            "defocus_blur",
            "elastic_transform",
            "fog",
            "frost",
            "gaussian_blur",
            "gaussian_noise",
            "glass_blur",
            "impulse_noise",
            "jpeg_compression",
            "motion_blur",
            "pixelate",
            "saturate",
            "shot_noise",
            "snow",
            "spatter",
            "speckle_noise",
            "zoom_blur",
        ]:
            raise ValueError(f"Invalid corruption type {self.corruption_type}. See documentation for options.")

        # Valid corruption severity
        if self.corruption_severity not in [1, 2, 3, 4, 5]:
            raise ValueError(f"Invalid severity {self.corruption_severity}. Options are from 1 to 5.")

        # Valid mode
        if self.mode not in ["train", "test"]:
            raise ValueError(f"Invalid mode {self.mode}. Options are 'train' and 'test.")


class ImageNetCDataset(Dataset):
    r"""
    ImageNet-C dataset from [1]_.

    It consits of 15 corruptions with 5 level of severity of the ImageNet validation set.
    Since there is no official train or test sets, we create a deterministic 80/20 split.

    Parameters
    ----------
    config: configuration class with
        save_dir: str
            Path from where to load the data.
        corruption_type: str, default="brightness"
            Type of corruption applied on the data.
        corruption_severity: int
            Severity of the corruption applied on the data.
        mode: str
            Mode. Options are "train" and "test".
        transform: Compose
            Transformation to apply to the images.

    Notes
    ----------
    ImageNet-C was created by Dan Hendrycks and Thomas Dietterich.
    Official website: https://github.com/hendrycks/robustness.

    Download
    --------
    To download the data, run the following command (replacing corruption by blur, digital, extra, noise, or weather):
    ```bash
    $ wget https://zenodo.org/record/2235448/files/corruption.tar
    ```
    Move the downloaded files to the desired location, e.g, datasets/imagenet_c, and extract them with
    ```bash
    $ tar -xvzf corruption.tar
    ```

    References
    ----------
    .. [1] D. Hendrycks and T. Dietterich. Benchmarking Neural Network Robustness to Common
           Corruptions and Surface Variations. In ICLR 2019
    """

    def __init__(self, config: ImageNetCDatasetConfig):
        super().__init__()
        train = True if config.mode == "train" else False

        # Recover dataset
        dataset = torchvision.datasets.ImageFolder(
            root=str(config.save_dir / config.corruption_type / str(config.corruption_severity))
        )
        samples = np.asarray(dataset.imgs)

        # Recover the (deterministic) train or test set
        indices = deterministic_split(data=samples, train_size=0.8, train=train)
        self.samples = samples[indices]
        self.n_classes = 1000

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
