from abc import ABC, abstractmethod

import numpy as np


class BaseMultimodalModel(ABC):
    r"""Abstract base class for multimodal models."""

    @abstractmethod
    def encode_image(self, images: list) -> np.ndarray: ...

    @abstractmethod
    def encode_text(self, sentences: list[str]) -> np.ndarray: ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...
