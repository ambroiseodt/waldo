from abc import ABC, abstractmethod

import numpy as np


class BaseVisionModel(ABC):
    r"""Abstract base class for vision models."""

    @abstractmethod
    def encode_image(self, images: list) -> np.ndarray:
        """Encode a list of images into embeddings."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings."""
        ...
