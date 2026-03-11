r"""
DINOv2 model following [1]_.

The model inherits from BaseVisionModel and its methods to recover
embeddings across layers and transformer models. See base.py for details.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension

References
----------
.. [1] M. Oquab et al. DINOv2: Learning Robust Visual Feature without Supervision. In TMLR 2024

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
from transformers import Dinov2Model, PreTrainedModel
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder, Dinov2MLP, Dinov2SwiGLUFFN

from ..utils import SwiGLUGateProjection, mean_pooling
from .base import BaseVisionModel, TransformerBlockAdapter

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Dinov2 configuration
# -------------------------------------------------------------------
@dataclass
class Dinov2Config:
    r"""
    Dinov2 configuration file.

    Parameters
    ----------
    model_size: str
        Type of model to use. Options are "base", "large", and "huge".
    patch_size: int
        Size of the patches to be extracted from the image.
    resolution: int
        Image resolution.
    device: str
        Device to use.
    """

    model_size: str = "base"
    patch_size: int = 16
    resolution: int = 224
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

        # Valid model size
        if self.resolution not in [224, 384]:
            raise ValueError(f"Invalid resolution {self.resolution}. Options are 224 and 384.")


# -------------------------------------------------------------------
# Dinov2 block adapter
# -------------------------------------------------------------------


class Dinov2BlockAdapter(TransformerBlockAdapter):
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

    Notes
    -----
    The standard implementation of DINOv2 uses
    For models with gated activations like SwiGLU, the feedforward layers are not necessarily linear anymore. For
    consistency, we define fc1 as the dense projection before the activation, act as the gated activation, and fc2 as
    the dense projection after the gate. Moreover, our implementation assumes that pre-normalization is used, that is,
    normalization layers are applied before the attention and feedforward blocks (and residual connections after). It is
    the standard implementation in modern transformers such as ViT, DINO or Llama models (see [1, 2]_ for details).
    It allows us to recover all modules (except rc1) using register_forward_hook, including the second residual
    connection, which is simply the output of the entire transformer block. For the first residual connection, it
    cannot be accessed as a forward, however it is the input to the second normalization layer and it can be accessed
    using register_forward_pre_hook instead of register_forward_hook to capture the input rather than the output of
    that submodule. Hence, our implementation is not tailored for models using post-norm and we advice users to rethink
    the design of the get_embedding method for their application as it might require changes.


    See also
    --------
    TransformerBlockAdapter
    """

    def _get_ln1(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.layer_scale1)

    def _get_mha(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.attention)

    def _get_rc1(self, layer: nn.Module) -> nn.Module:
        # Use pre_hook to capture rc1
        return cast(nn.Module, layer.layer_scale2)

    def _get_ln2(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.layer_scale2)

    def _get_fc1(self, layer: nn.Module) -> nn.Module:
        if isinstance(layer.mlp, Dinov2SwiGLUFFN):
            return cast(nn.Module, SwiGLUGateProjection(layer.mlp.weights_in))
        if isinstance(layer.mlp, Dinov2MLP):
            return cast(nn.Module, layer.mlp.fc1)
        raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected Dinov2SwiGLUFFN or Dinov2MLP.")

    def _get_act(self, layer: nn.Module) -> nn.Module:
        if isinstance(layer.mlp, Dinov2SwiGLUFFN):
            # Use pre_hook to capture act
            return cast(nn.Module, layer.mlp.fc2)
        if isinstance(layer.mlp, Dinov2MLP):
            return cast(nn.Module, layer.mlp.activation)
        raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected Dinov2SwiGLUFFN or Dinov2MLP.")

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        if isinstance(layer.mlp, Dinov2SwiGLUFFN):
            return cast(nn.Module, SwiGLUGateProjection(layer.mlp.weights_out))
        if isinstance(layer.mlp, Dinov2MLP):
            return cast(nn.Module, layer.mlp.fc2)
        raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected Dinov2SwiGLUFFN or Dinov2MLP.")

    def _get_rc2(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.output)

    def _get_pre_hooks_modules(self, layer: nn.Module) -> set[str]:
        r"""
        Returns the names of the module for which register_forward_pre_hook should be used
        in place of register_forward_hook to access their inputs instead of the outputs.
        """
        if isinstance(layer.mlp, Dinov2SwiGLUFFN):
            return {"rc1", "act"}
        if isinstance(layer.mlp, Dinov2MLP):
            return {"rc1", "act"}
        raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected Dinov2SwiGLUFFN or Dinov2MLP.")


# -------------------------------------------------------------------
# Dinov2 model
# -------------------------------------------------------------------
class Dinov2(BaseVisionModel):
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

    def __init__(self, config: Dinov2Config):
        super().__init__()
        self.device = config.device
        self.model_name = f"facebook/dinov2-{config.model_size.lower()}"
        self.model = self._load_from_huggingface(model_name=self.model_name, device=self.device)

    def _load_from_huggingface(self, model_name: str, device: str) -> PreTrainedModel:
        r"""
        Load weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "facebook/dinov2-small",
                "facebook/dinov2-base",
                "facebook/dinov2-large",
                "facebook/dinov2-giant".
        device: str
            Device to use.

        Notes
        -----
        The model is set to evaluation mode after loading (dropout disabled).
        Call model.train() explicitly to re-enable training mode.
        """
        available_models = [
            "facebook/dinov2-small",
            "facebook/dinov2-base",
            "facebook/dinov2-large",
            "facebook/dinov2-giant",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")
        model = Dinov2Model.from_pretrained(f"{model_name}", device_map=device)
        model.eval()
        return model

    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Mean pooling over patch tokens stored on cpu. Since the embeddings are obtained
        from get_embeddings that uses @torch_inference, there is no need to detach the tensor.

        Parameters
        ----------
        x: torch.Tensor of shape (N, L, E)
            Sequence of tokens to pool.

        Returns
        -------
        emb: torch.Tensor of shape (N, E)
            Pooled embedding stored on CPU.
        """
        return mean_pooling(x=x, exclude_cls=True).cpu()

    @property
    def block_adapter(self) -> type[TransformerBlockAdapter]:
        r"""Return the block adapter class for the model's transformer blocks."""
        return Dinov2BlockAdapter

    @property
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        encoder = cast(Dinov2Encoder, self.model.encoder)
        return cast(ModuleList, encoder.layer)
