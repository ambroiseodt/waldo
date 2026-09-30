r"""
AIMv2 model following [1]_. Our implementation allows to study specific layers and modules.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension

Notes
-----
We focus on the pretrained models available on HuggingFace for image feature extraction.
See https://huggingface.co/collections/apple/aimv2 for other model variants.

References
----------
.. [1] E. Fini et al. Multimodal Autoregressive Pre-training of Large Vision Encoders. In CVPR 2025

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

import torch.nn as nn
from torch.nn import ModuleList
from transformers import Aimv2VisionModel, PreTrainedModel
from transformers.models.aimv2.modeling_aimv2 import Aimv2Encoder, Aimv2MLP

from ..transformer_block import TransformerBlockAdapter
from .base import BaseVisionModel

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
#   Config
# -------------------------------------------------------------------
@dataclass
class Aimv2Config:
    r"""
    AIMv2 configuration file.

    Parameters
    ----------
    model_size: str
        Type of model to use. Options are "large", "huge", "1B", and "3B".
    resolution: int
        Image resolution. Options are 224, 336, 448.
    pooling: str
        Pooling over the sequence of tokens. Only "content" (mean over all tokens) is available since
        AIMv2 has no CLS token.
    device: str
        Device to use.
    """

    model_size: str = "large"
    resolution: int = 224
    pooling: str = "content"
    device: str = "cpu"

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Valid model size
        if self.model_size not in ["large", "huge", "1B", "3B"]:
            raise ValueError(f"Invalid model size {self.model_size}. Options are 'large', 'huge', '1B', and '3B'.")

        # Valid resolution
        if self.resolution not in [224, 336, 448]:
            raise ValueError(f"Invalid resolution {self.resolution}. Options are 224, 336 and 448.")

        # Valid pooling
        if self.pooling != "content":
            raise ValueError(f"Invalid pooling {self.pooling}. AIMv2 has no CLS token, the only option is 'content'.")


# -------------------------------------------------------------------
#   Transformer block adapter
# -------------------------------------------------------------------
class Aimv2BlockAdapter(TransformerBlockAdapter):
    r"""
    Adapter mapping the standard module names to the submodules of a AIMv2 transformer block.

    Parameters
    ----------
    layer: nn.Module
        A single AIMv2 transformer block.

    Notes
    -----
    AIMv2 relies on RMSNorm and on a SwiGLU activation in the FFN [1, 2, 3]_: act is the input of the down projection,
    following the decomposition of TransformerBlockAdapter.

    References
    ----------
    .. [1] P. Ramachandran et al. Searching for activation functions. arXiv 2017

    .. [2] Y. Dauphin et al. Language Modeling with Gated Convolutional Networks. In ICML 2017

    .. [3] N. Shazeer. GLU Variants Improve Transformer. arXiv 2020

    See also
    --------
    TransformerBlockAdapter
    """

    def _get_ln1(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.rms_norm1)

    def _get_mha(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.attention)

    def _get_rc1(self, layer: nn.Module) -> nn.Module:
        # Use pre_hook to capture rc1
        return cast(nn.Module, layer.rms_norm2)

    def _get_ln2(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.rms_norm2)

    def _get_fc1(self, layer: nn.Module) -> nn.Module:
        ffn = cast(Aimv2MLP, layer.ffn)
        return cast(nn.Module, ffn.gate_proj)

    def _get_act(self, layer: nn.Module) -> nn.Module:
        # Use pre_hook to capture act
        ffn = cast(Aimv2MLP, layer.ffn)
        return cast(nn.Module, ffn.down_proj)

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        ffn = cast(Aimv2MLP, layer.ffn)
        return cast(nn.Module, ffn.down_proj)

    def _get_rc2(self, layer: nn.Module) -> nn.Module:
        return layer

    def _get_pre_hooks_modules(self, layer: nn.Module) -> set[str]:
        r"""
        Returns the names of the modules for which register_forward_pre_hook should be used
        in place of register_forward_hook to access their inputs instead of their outputs.
        """
        return {"rc1", "act"}


# -------------------------------------------------------------------
#   Model
# -------------------------------------------------------------------
class Aimv2(BaseVisionModel):
    r"""
    AIMv2 implementation following [1]_.

    Parameters
    ----------
    config: configuration class with
        model_size: str
            Type of model to use. Options are "large", "huge", "1B", and "3B".
        resolution: int
            Image resolution. Options are 224, 336, 448.
        pooling: str
            Pooling over the sequence of tokens. Only "content" (mean over all tokens) is available since
            AIMv2 has no CLS token.
        device: str
            Device to use.

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled). Call `model.train()` explicitly
    if you intend to fine-tune.

    References
    ----------
    .. [1] E. Fini et al. Multimodal Autoregressive Pre-training of Large Vision Encoders. In CVPR 2025
    """

    def __init__(self, config: Aimv2Config):
        super().__init__()
        self.device = config.device
        self.pooling_type = config.pooling
        self.model_name = f"apple/aimv2-{config.model_size}-patch14-{config.resolution}"
        self.model = self._load_model(model_name=self.model_name, device=self.device)

    def _load_model(self, model_name: str, device: str) -> PreTrainedModel:
        r"""
        Load weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "apple/aimv2-large-patch14-224",
                "apple/aimv2-large-patch14-336",
                "apple/aimv2-large-patch14-448",
                "apple/aimv2-huge-patch14-224",
                "apple/aimv2-huge-patch14-336",
                "apple/aimv2-huge-patch14-448",
                "apple/aimv2-1B-patch14-224",
                 "apple/aimv2-1B-patch14-336",
                 "apple/aimv2-1B-patch14-448",
                "apple/aimv2-3B-patch14-224",
                "apple/aimv2-3B-patch14-336",
                "apple/aimv2-3B-patch14-448".
        device: str
            Device to use.

        Notes
        -----
        The model is loaded in evaluation mode by default (dropout disabled). Call `model.train()` explicitly
        if you intend to fine-tune.
        """
        available_models = [
            "apple/aimv2-large-patch14-224",
            "apple/aimv2-large-patch14-336",
            "apple/aimv2-large-patch14-448",
            "apple/aimv2-huge-patch14-224",
            "apple/aimv2-huge-patch14-336",
            "apple/aimv2-huge-patch14-448",
            "apple/aimv2-1B-patch14-224",
            "apple/aimv2-1B-patch14-336",
            "apple/aimv2-1B-patch14-448",
            "apple/aimv2-3B-patch14-224",
            "apple/aimv2-3B-patch14-336",
            "apple/aimv2-3B-patch14-448",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")
        model = Aimv2VisionModel.from_pretrained(f"{model_name}", device_map=device)
        model.eval()
        return model

    @property
    def block_adapter(self) -> type[TransformerBlockAdapter]:
        r"""Return the block adapter class for the model's transformer blocks."""
        return Aimv2BlockAdapter

    @property
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        encoder = cast(Aimv2Encoder, self.model.encoder)
        return cast(ModuleList, encoder.layers)

    @property
    def final_norm(self) -> nn.Module:
        r"""Return the norm applied to the output of the last transformer block."""
        return cast(nn.Module, self.model.rms_norm)

    @property
    def n_prefix_tokens(self) -> int:
        r"""Return the number of auxiliary tokens. AIMv2 has no CLS token nor register tokens."""
        return 0
