r"""
Vision Transformer model pretrained on ImageNet-21k following [1]_. It allows to study specific layers and modules.


Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension

Notes
-----
We focus on the pretrained models available on HuggingFace for image feature extraction.
See https://huggingface.co/google/models for other model variants.

References
----------
.. [1] A. Dosovitskiy, L. Beyer et al. An Image is Worth 16x16 Words: Transformers
       for Image Recognition at Scale. In ICLR 2021

See also
--------
TransformerBlockAdapter
BaseVisionModel

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch.nn import ModuleList
from transformers import PreTrainedModel, ViTModel
from transformers.models.vit.modeling_vit import ViTEncoder, ViTIntermediate, ViTOutput

from ..transformer_block import TransformerBlockAdapter
from ..utils import mean_pooling
from .base import BaseVisionModel

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
#   Config
# -------------------------------------------------------------------
@dataclass
class ViTConfig:
    r"""
    ViT configuration file.

    Parameters
    ----------
    model_size: str
        Type of model to use. Options are "base", "large", and "huge".
    patch_size: int
        Size of the patches to be extracted from the image. Options are 14, 16 and 32.
    device: str
        Device to use.
    """

    model_size: str = "base"
    patch_size: int = 16
    device: str = "cpu"

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Valid model size
        if self.model_size not in ["base", "large", "huge"]:
            raise ValueError(f"Invalid model size {self.model_size}. Options are 'base', 'large' and 'huge'.")

        # Valid patch size
        if self.patch_size not in [14, 16, 32]:
            raise ValueError(f"Invalid patch size {self.patch_size}. Options are 14, 16 and 32.")


# -------------------------------------------------------------------
#   Transformer block adapter
# -------------------------------------------------------------------
class ViTBlockAdapter(TransformerBlockAdapter):
    r"""
    Adapter that maps standard module names to a transformer block's submodules.
    Subclasses need to implement the 8 methods that outputs the following modules:
        ln1: first layer norm
        mha: multi-head attention
        rc1: first residual connection
        ln2: second layer norm
        fc1: first feedforward linear layer
        act: feedforward activation function
        fc2: second feedforward linear layer
        rc2: second residual connection

    Parameters
    ----------
    layer: nn.Module
        A single transformer block.


    See also
    --------
    TransformerBlockAdapter
    """

    def _get_ln1(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.layernorm_before)

    def _get_mha(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.attention)

    def _get_rc1(self, layer: nn.Module) -> nn.Module:
        # Use pre_hook to capture rc1
        return cast(nn.Module, layer.layernorm_after)

    def _get_ln2(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.layernorm_after)

    def _get_fc1(self, layer: nn.Module) -> nn.Module:
        intermediate = cast(ViTIntermediate, layer.intermediate)
        return cast(nn.Module, intermediate.dense)

    def _get_act(self, layer: nn.Module) -> nn.Module:
        intermediate = cast(ViTIntermediate, layer.intermediate)
        return cast(nn.Module, intermediate.intermediate_act_fn)

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        output = cast(ViTOutput, layer.output)
        return cast(nn.Module, output.dense)

    def _get_rc2(self, layer: nn.Module) -> nn.Module:
        return layer

    def _get_pre_hooks_modules(self, layer: nn.Module) -> set[str]:
        r"""
        Returns the names of the module for which register_forward_pre_hook should be used
        in place of register_forward_hook to access their inputs instead of the outputs.
        """
        return {"rc1"}


# -------------------------------------------------------------------
#   Model
# -------------------------------------------------------------------
class ViT(BaseVisionModel):
    r"""
    Vision Transformer implementation following [1]_.


    Parameters
    ----------
    config: configuration class with
        model_size: str
            Type of model to use. Options are "base", "large", and "huge".
        patch_size: int
            Size of the patches to be extracted from the image.
        resolution: int
            Image resolution.
        device: str
            Device to use.

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled,
    BatchNorm uses running statistics). Call `model.train()` explicitly
    if you intend to fine-tune.

    References
    ----------
    .. [1] A. Dosovitskiy, L. Beyer et al. An Image is Worth 16x16 Words: Transformers
           for Image Recognition at Scale. ICLR 2021.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.device = config.device
        self.model_name = f"google/vit-{config.model_size}-patch{config.patch_size}-224-in21k"
        self.model = self._load_model(model_name=self.model_name, device=self.device)

    def _load_model(self, model_name: str, device: str) -> PreTrainedModel:
        r"""
        Load weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "google/vit-base-patch16-224-in21k",
                "google/vit-base-patch32-224-in21k",
                "google/vit-large-patch16-224-in21k",
                "google/vit-large-patch32-224-in21k",
                "google/vit-huge-patch14-224-in21k".

        device: str
            Device to use.

        Notes
        -----
        The model is loaded in evaluation mode by default (dropout disabled,
        BatchNorm uses running statistics). Call `model.train()` explicitly
        if you intend to fine-tune.
        """
        available_models = [
            "google/vit-base-patch16-224-in21k",
            "google/vit-base-patch32-224-in21k",
            "google/vit-large-patch16-224-in21k",
            "google/vit-large-patch32-224-in21k",
            "google/vit-huge-patch14-224-in21k",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")
        model = ViTModel.from_pretrained(f"{model_name}", device_map=device)
        model.eval()
        return model

    def pooling(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Mean pooling over patch tokens stored on cpu. Since the embeddings are obtained
        from get_embeddings that uses @torch_inference, there is no need to detach the tensor.

        In ViT, there is a CLS token, so we need to exclude it before the global pooling.

        Parameters
        ----------
        hidden_states: torch.Tensor of shape (N, L, E)
            Sequence of tokens to pool.

        Returns
        -------
        torch.Tensor of shape (N, E)
            Pooled embedding stored on CPU.
        """
        return mean_pooling(hidden_states=hidden_states, exclude_cls=True).cpu()

    @property
    def block_adapter(self) -> type[TransformerBlockAdapter]:
        r"""Return the block adapter class for the model's transformer blocks."""
        return ViTBlockAdapter

    @property
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        encoder = cast(ViTEncoder, self.model.encoder)
        return cast(ModuleList, encoder.layer)
