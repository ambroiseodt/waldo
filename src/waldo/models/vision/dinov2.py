r"""
DINOv2 model following [1]_, with or without register tokens [2]_. It allows to study specific layers and modules.

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
.. [1] M. Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. In TMLR 2024

.. [2] T. Darcet et al. Vision Transformers Need Registers. In ICLR 2024

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
from transformers import Dinov2Model, Dinov2WithRegistersModel, PreTrainedModel
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder, Dinov2MLP, Dinov2SwiGLUFFN
from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersMLP,
    Dinov2WithRegistersSwiGLUFFN,
)

from ..transformer_block import TransformerBlockAdapter
from .base import BaseVisionModel

logger = logging.getLogger(__name__)

# FFN types of DINOv2 with and without registers
SWIGLU_FFN = (Dinov2SwiGLUFFN, Dinov2WithRegistersSwiGLUFFN)
MLP_FFN = (Dinov2MLP, Dinov2WithRegistersMLP)


# -------------------------------------------------------------------
#   Config
# -------------------------------------------------------------------
@dataclass
class Dinov2Config:
    r"""
    DINOv2 configuration file.

    Parameters
    ----------
    model_size: str
        Type of model to use. Options are "small", "base", "large", and "giant", and their variants trained
        with register tokens [1]_ "small-registers", "base-registers", "large-registers", and "giant-registers".
    pooling: str
        Pooling over the sequence of tokens. Options are "content" (mean over the patch tokens, default)
        and "all" (concatenation of the CLS token and the mean over the patch tokens).
    device: str
        Device to use.

    References
    ----------
    .. [1] T. Darcet et al. Vision Transformers Need Registers. In ICLR 2024
    """

    model_size: str = "base"
    pooling: str = "content"
    device: str = "cpu"

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Valid model size
        sizes = ["small", "base", "large", "giant"]
        valid_sizes = sizes + [f"{size}-registers" for size in sizes]
        if self.model_size not in valid_sizes:
            raise ValueError(f"Invalid model size {self.model_size}. Options are {valid_sizes}.")

        # Valid pooling
        if self.pooling not in ["content", "all"]:
            raise ValueError(f"Invalid pooling {self.pooling}. Options are 'content' and 'all'.")


# -------------------------------------------------------------------
#   Transformer block adapter
# -------------------------------------------------------------------
class Dinov2BlockAdapter(TransformerBlockAdapter):
    r"""
    Adapter mapping the standard module names to the submodules of a DINOv2 transformer block.

    Parameters
    ----------
    layer: nn.Module
        A single DINOv2 transformer block.

    Notes
    -----
    The FFN of DINOv2 is an MLP with a GELU activation, except for DINOv2-g which relies on a SwiGLU activation
    [1, 2, 3]_, with or without registers. In HuggingFace, the gate and up projections of the SwiGLU FFN are fused in
    a single linear layer (weights_in): fc1 is the gate projection recovered from its output with process_output, and
    act is the input of the down projection (weights_out), following the decomposition of TransformerBlockAdapter.
    DINOv2 also scales the outputs of the attention and FFN blocks with LayerScale [4]_ before the residual
    connections. For consistency with other models, we do not probe them.

    References
    ----------
    .. [1] P. Ramachandran et al. Searching for activation functions. arXiv 2017

    .. [2] Y. Dauphin et al. Language Modeling with Gated Convolutional Networks. In ICML 2017

    .. [3] N. Shazeer. GLU Variants Improve Transformer. arXiv 2020

    .. [4] H. Touvron et al. Going Deeper With Image Transformers. In ICCV 2021

    See also
    --------
    TransformerBlockAdapter
    """

    def __init__(self, layer: nn.Module):
        self.is_swiglu = isinstance(layer.mlp, SWIGLU_FFN)
        super().__init__(layer)

    def _get_ln1(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.norm1)

    def _get_mha(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.attention)

    def _get_rc1(self, layer: nn.Module) -> nn.Module:
        # Use pre_hook to capture rc1
        return cast(nn.Module, layer.norm2)

    def _get_ln2(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.norm2)

    def _get_fc1(self, layer: nn.Module) -> nn.Module:
        r"""
        Returns the native FC1 projection module.

        For SwiGLU blocks, ``weights_in`` produces the concatenated gate and
        up projections; ``process_output`` extracts the gate projection used
        as FC1 for probing.

        See also
        --------
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py
        """
        if isinstance(layer.mlp, SWIGLU_FFN):
            # weights_in outputs [gate, up]; process_output extracts the gate as FC1.
            return cast(nn.Module, layer.mlp.weights_in)
        elif isinstance(layer.mlp, MLP_FFN):
            return cast(nn.Module, layer.mlp.fc1)
        else:
            raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected a SwiGLU FFN or an MLP.")

    def _get_act(self, layer: nn.Module) -> nn.Module:
        # Use pre_hook to capture act
        if isinstance(layer.mlp, SWIGLU_FFN):
            return cast(nn.Module, layer.mlp.weights_out)
        # Use hook to capture act
        elif isinstance(layer.mlp, MLP_FFN):
            return cast(nn.Module, layer.mlp.activation)
        else:
            raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected a SwiGLU FFN or an MLP.")

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        if isinstance(layer.mlp, SWIGLU_FFN):
            return cast(nn.Module, layer.mlp.weights_out)
        elif isinstance(layer.mlp, MLP_FFN):
            return cast(nn.Module, layer.mlp.fc2)
        else:
            raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected a SwiGLU FFN or an MLP.")

    def _get_rc2(self, layer: nn.Module) -> nn.Module:
        return layer

    def _get_pre_hooks_modules(self, layer: nn.Module) -> set[str]:
        r"""
        Returns the names of the modules for which register_forward_pre_hook should be used
        in place of register_forward_hook to access their inputs instead of their outputs.
        """
        if isinstance(layer.mlp, SWIGLU_FFN):
            return {"rc1", "act"}
        elif isinstance(layer.mlp, MLP_FFN):
            return {"rc1"}
        else:
            raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected a SwiGLU FFN or an MLP.")

    def process_output(self, module_name: str, output: torch.Tensor) -> torch.Tensor:
        r"""
        Extracts the gate projection from the concatenated SwiGLU ``weights_in`` output.
        """
        if module_name == "fc1" and self.is_swiglu:
            return output.chunk(2, dim=-1)[0]
        return output


# -------------------------------------------------------------------
#   Model
# -------------------------------------------------------------------
class Dinov2(BaseVisionModel):
    r"""
    DINOv2 implementation following [1]_, with or without register tokens [2]_.

    Parameters
    ----------
    config: configuration class with
        model_size: str
            Type of model to use. Options are "small", "base", "large", and "giant", and their variants
            trained with register tokens "small-registers", "base-registers", "large-registers", and
            "giant-registers".
        pooling: str
            Pooling over the sequence of tokens. Options are "content" (mean over the patch tokens, default)
            and "all" (concatenation of the CLS token and the mean over the patch tokens).
        device: str
            Device to use.

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled). Call `model.train()` explicitly
    if you intend to fine-tune.

    References
    ----------
    .. [1] M. Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. In TMLR 2024

    .. [2] T. Darcet et al. Vision Transformers Need Registers. In ICLR 2024
    """

    def __init__(self, config: Dinov2Config):
        super().__init__()
        self.device = config.device
        self.pooling_type = config.pooling
        # Models with registers are named facebook/dinov2-with-registers-{size} on HuggingFace
        size, registers = config.model_size.removesuffix("-registers"), config.model_size.endswith("-registers")
        prefix = "dinov2-with-registers" if registers else "dinov2"
        self.model_name = f"facebook/{prefix}-{size}"
        self.model = self._load_model(model_name=self.model_name, device=self.device)

    def _load_model(self, model_name: str, device: str) -> PreTrainedModel:
        r"""
        Load weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "facebook/dinov2-small",
                "facebook/dinov2-base",
                "facebook/dinov2-large",
                "facebook/dinov2-giant",
                "facebook/dinov2-with-registers-small",
                "facebook/dinov2-with-registers-base",
                "facebook/dinov2-with-registers-large",
                "facebook/dinov2-with-registers-giant".
        device: str
            Device to use.

        Notes
        -----
        The model is loaded in evaluation mode by default (dropout disabled). Call `model.train()` explicitly
        if you intend to fine-tune.
        """
        available_models = [
            "facebook/dinov2-small",
            "facebook/dinov2-base",
            "facebook/dinov2-large",
            "facebook/dinov2-giant",
            "facebook/dinov2-with-registers-small",
            "facebook/dinov2-with-registers-base",
            "facebook/dinov2-with-registers-large",
            "facebook/dinov2-with-registers-giant",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")
        model_type = Dinov2WithRegistersModel if "with-registers" in model_name else Dinov2Model
        model = model_type.from_pretrained(f"{model_name}", device_map=device)
        model.eval()
        return model

    @property
    def block_adapter(self) -> type[TransformerBlockAdapter]:
        r"""Return the block adapter class for the model's transformer blocks."""
        return Dinov2BlockAdapter

    @property
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        encoder = cast(Dinov2Encoder, self.model.encoder)
        return cast(ModuleList, encoder.layer)

    @property
    def final_norm(self) -> nn.Module:
        r"""Return the norm applied to the output of the last transformer block."""
        return cast(nn.Module, self.model.layernorm)

    @property
    def n_prefix_tokens(self) -> int:
        r"""Return the number of auxiliary tokens, i.e., the CLS token followed by the register tokens if any."""
        return 1 + getattr(self.model.config, "num_register_tokens", 0)
