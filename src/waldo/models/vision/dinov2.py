r"""
DINOv2 model following [1]_. It allows to study specific layer and modules.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension

Notes
-----
We focus on the pretrained models available on HuggingFace for image feature extraction.
See https://huggingface.co/collections/facebook/dinov2 for other model variants.

References
----------
.. [1] M. Oquab et al. DINOv2: Learning Robust Visual Feature without Supervision. In TMLR 2024

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
from transformers import Dinov2Model, PreTrainedModel
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder, Dinov2MLP, Dinov2SwiGLUFFN

from ..transformer_block import TransformerBlockAdapter
from ..utils import SwiGLUGateProjection, mean_pooling
from .base import BaseVisionModel

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
#   Config
# -------------------------------------------------------------------
@dataclass
class Dinov2Config:
    r"""
    Dinov2 configuration file.

    Parameters
    ----------
    model_size: str
        Type of model to use. Options are "small", "base", "large", and "giant".
    device: str
        Device to use.
    """

    model_size: str = "base"
    device: str = "cpu"

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Valid model size
        if self.model_size not in ["small", "base", "large", "giant"]:
            raise ValueError(f"Invalid model size {self.model_size}. Options are 'small', 'base', 'large' and 'giant'.")


# -------------------------------------------------------------------
#   Transformer block adapter
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
    DINOv2-g model relies on a gated SwiGLU activation in the FFN (see [1, 2]_ for details). For consistency
    with ViT, we decompose the FFN block as follows:

        fc1  = gate_proj(x)                      -> linear layer which feeds the activation
        act  = silu(gate_proj(x)) * up_proj(x)   -> non-linearity (which is a function of the input too)
        fc2  = down_proj(act)                    -> linear layer which gives the output of the FFN

    This ensures that fc1 and fc2 remain pure linear operations, and the gating mechanism is absorbed into act. The
    up_proj can be interpreted as part of the activation definition rather than a separate linear layer, making act
    the complete sequence-to-sequence non-linearity between fc1 and fc2. Subclasses must reflect this convention in
    their _get_fc1, _get_act, and _get_fc2 implementations, and declare act as a pre-hook module in _pre_hook_modules
    when SwiGLU is used, since the gated activation output is recovered as the input to down_proj. Note that DINOv2
    makes use of slight modifications of the transformer architecture (see, e.g., LayerScale [3]_) that do not
    fundamentally change the transformer architecture and merely improve memory-efficiency and training
    stability. For the consistency of our analysis, we do not probe them.

    References
    ----------
    .. [1] Y. Dauphin et al. Language Modeling with Gated Convolutional Networks. In ICML 2017

    .. [2] N. Shazeer. GLU Variants Improve Transformer. arXiv 2020

    .. [3] H. Touvron et al. Going Deeper With Image Transformers. ICCV 2021

    See also
    --------
    TransformerBlockAdapter
    """

    def _get_ln1(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.norm1)

    def _get_mha(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.attention)

    def _get_rc1(self, layer: nn.Module) -> nn.Module:
        # Use pre_hook to capture rc1
        return cast(nn.Module, layer.norm1)

    def _get_ln2(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.norm1)

    def _get_fc1(self, layer: nn.Module) -> nn.Module:
        r"""
        When SwiGLUFFN is used, the DINOv2 implementation concatenates the gate_proj and up_proj
        using a matrix weight_in of shape emb_dim x 2 * hidden_dim and then recover gate_proj
        and up_proj by chunk. The SwiGLUGateProjection allows us to recover only the gate_proj to
        obtain fc1.

        See also
        --------
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py
        and SwiGLUGateProjection in models/utils.py
        """
        if isinstance(layer.mlp, Dinov2SwiGLUFFN):
            return cast(nn.Module, SwiGLUGateProjection(layer.mlp.weights_in))
        elif isinstance(layer.mlp, Dinov2MLP):
            return cast(nn.Module, layer.mlp.fc1)
        else:
            raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected Dinov2SwiGLUFFN or Dinov2MLP.")

    def _get_act(self, layer: nn.Module) -> nn.Module:
        # Use pre_hook to capture act
        if isinstance(layer.mlp, Dinov2SwiGLUFFN):
            return cast(nn.Module, layer.mlp.weights_out)
        # Use hook to capture act
        elif isinstance(layer.mlp, Dinov2MLP):
            return cast(nn.Module, layer.mlp.activation)
        else:
            raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected Dinov2SwiGLUFFN or Dinov2MLP.")

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        if isinstance(layer.mlp, Dinov2SwiGLUFFN):
            return cast(nn.Module, layer.mlp.weights_out)
        elif isinstance(layer.mlp, Dinov2MLP):
            return cast(nn.Module, layer.mlp.fc2)
        else:
            raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected Dinov2SwiGLUFFN or Dinov2MLP.")

    def _get_rc2(self, layer: nn.Module) -> nn.Module:
        return layer

    def _get_pre_hooks_modules(self, layer: nn.Module) -> set[str]:
        r"""
        Returns the names of the module for which register_forward_pre_hook should be used
        in place of register_forward_hook to access their inputs instead of the outputs.
        """
        if isinstance(layer.mlp, Dinov2SwiGLUFFN):
            return {"rc1", "act"}
        elif isinstance(layer.mlp, Dinov2MLP):
            return {"rc1"}
        else:
            raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected Dinov2SwiGLUFFN or Dinov2MLP.")


# -------------------------------------------------------------------
#   Model
# -------------------------------------------------------------------
class Dinov2(BaseVisionModel):
    r"""
    DINOv2 implementation following [1]_.


    Parameters
    ----------
    config: configuration class with
        model_size: str
            Type of model to use. Options are "small", "base", "large", and "giant".

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled,
    BatchNorm uses running statistics). Call `model.train()` explicitly
    if you intend to fine-tune.

    References
    ----------
    .. [1] M. Oquab et al. DINOv2: Learning Robust Visual Feature without Supervision. In TMLR 2024
    """

    def __init__(self, config: Dinov2Config):
        super().__init__()
        self.device = config.device
        self.model_name = f"facebook/dinov2-{config.model_size}"
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

    def pooling(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Mean pooling over patch tokens stored on cpu. Since the embeddings are obtained
        from get_embeddings that uses @torch_inference, there is no need to detach the tensor.

        In DINOv2, there is a CLS token, so we need to exclude it before the global pooling.

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
        return Dinov2BlockAdapter

    @property
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        encoder = cast(Dinov2Encoder, self.model.encoder)
        return cast(ModuleList, encoder.layer)
