r"""
iWildCam Dataset.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.transforms import Compose
from wilds import get_dataset

from ...infra import DATASET_DIR


# ------------------------------------------------------------------------------
#   Config
# ------------------------------------------------------------------------------
@dataclass
class IWildCamDatasetConfig:
    r"""
    iWildCam configuration file.

    Parameters
    ----------
    save_dir: Path
        Path from where to load the data or save them if it does not exist.
    mode: str
        Mode. Options are "train" and "test".
    transform: Compose | None
        Transformation to apply to the images.
    """

    save_dir: Path = DATASET_DIR / "iwildcam"
    mode: str = "train"
    transform: Compose | None = None

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        valid_modes = ["train", "test"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Options are {valid_modes}.")


# ------------------------------------------------------------------------------
#   Dataset
# ------------------------------------------------------------------------------
class IWildCamDataset(Dataset):
    r"""
    iWildCam dataset from [1]_, part of the WILDS benchmark [2]_.

    iWildCam consists of camera-trap photographs of wildlife, labeled with one of
    182 animal species (including "no animal"), collected from hundreds of camera
    trap locations. The distribution shift occurs across locations, which differ in
    background, lighting, and camera angle.

    Parameters
    ----------
    config: configuration class with
        save_dir: Path
            Path from where to load the data or save them if it does not exist.
        mode: str
            Mode. Options are "train" and "test".
        transform: Compose
            Transformation to apply to the images.

    Notes
    -----
    iWildCam was created by the iWildCam challenge organizers, adapted for domain
    generalization benchmarking by the WILDS team.
    Official website: https://wilds.stanford.edu/datasets/#iwildcam

    References
    ----------
    .. [1] S. Beery et al. The iWildCam 2020 Competition Dataset. arXiv 2020
    .. [2] P. W. Koh et al. WILDS: A Benchmark of in-the-Wild Distribution Shifts. ICML 2021
    """

    def __init__(self, config: IWildCamDatasetConfig):
        super().__init__()
        full_dataset = get_dataset(dataset="iwildcam", download=True, root_dir=str(config.save_dir))
        assert full_dataset is not None
        self.samples = full_dataset.get_subset(config.mode)
        self.n_classes = 182

        # Recover transform
        self.transform = config.transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        # WILDS subsets return (image, label, metadata)
        sample, label = self.samples[idx][:2]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __repr__(self):
        return f"Dataset with {len(self.samples)} images."
