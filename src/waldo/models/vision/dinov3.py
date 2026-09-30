r"""
DINOv3 ViT models pretrained on LVD-1689M following [1]_. It allows to study specific layers and modules.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension

Notes
-----
We focus on the pretrained models available on HuggingFace for image feature extraction.
See https://huggingface.co/collections/facebook/dinov3 for other model variants.

References
----------
.. [1] O. Simeoni et al. DINOv3. arXiv 2025

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
from transformers import DINOv3ViTModel, PreTrainedModel
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTEncoder, DINOv3ViTGatedMLP, DINOv3ViTMLP

from ..transformer_block import TransformerBlockAdapter
from .base import BaseVisionModel

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
#   Config
# -------------------------------------------------------------------
@dataclass
class Dinov3Config:
    r"""
    DINOv3 configuration file.

    Parameters
    ----------
    model_size: str
        Type of model to use. Options are "vits16", "vitb16", "vitl16", "vit7b16", "vits16plus" and "vith16plus".
    pooling: str
        Pooling over the sequence of tokens. Options are "content" (mean over the patch tokens, default)
        and "all" (concatenation of the CLS token and the mean over the patch tokens).
    device: str
        Device to use.
    """

    model_size: str = "vitb16"
    pooling: str = "content"
    device: str = "cpu"

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Valid model size
        if self.model_size not in ["vits16", "vitb16", "vitl16", "vit7b16", "vits16plus", "vith16plus"]:
            raise ValueError(
                f"Invalid model size {self.model_size}."
                " Options are 'vits16', 'vitb16', 'vitl16', 'vit7b16', 'vits16plus' and 'vith16plus'."
            )

        # Valid pooling
        if self.pooling not in ["content", "all"]:
            raise ValueError(f"Invalid pooling {self.pooling}. Options are 'content' and 'all'.")


# -------------------------------------------------------------------
#   Transformer block adapter
# -------------------------------------------------------------------
class Dinov3BlockAdapter(TransformerBlockAdapter):
    r"""
    Adapter mapping the standard module names to the submodules of a DINOv3 transformer block.

    Parameters
    ----------
    layer: nn.Module
        A single DINOv3 transformer block.

    Notes
    -----
    The FFN of DINOv3 is an MLP with a GELU activation for ViT-S, ViT-B and ViT-L, and a gated MLP with a SwiGLU
    activation [1, 2, 3]_ for ViT-S+, ViT-H+ and ViT-7B. For the latter, act is the input of the down projection,
    following the decomposition of TransformerBlockAdapter. DINOv3 also scales the outputs of the attention and FFN
    blocks with LayerScale [4]_ before the residual connections. For consistency with other models, we do not probe
    them.

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
        if isinstance(layer.mlp, DINOv3ViTGatedMLP):
            return cast(nn.Module, layer.mlp.gate_proj)
        elif isinstance(layer.mlp, DINOv3ViTMLP):
            return cast(nn.Module, layer.mlp.up_proj)
        else:
            raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected DINOv3ViTGatedMLP or DINOv3ViTMLP.")

    def _get_act(self, layer: nn.Module) -> nn.Module:
        # Use pre_hook to capture act
        if isinstance(layer.mlp, DINOv3ViTGatedMLP):
            return cast(nn.Module, layer.mlp.down_proj)
        # Use hook to capture act
        elif isinstance(layer.mlp, DINOv3ViTMLP):
            return cast(nn.Module, layer.mlp.act_fn)
        else:
            raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected DINOv3ViTGatedMLP or DINOv3ViTMLP.")

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        if isinstance(layer.mlp, (DINOv3ViTGatedMLP, DINOv3ViTMLP)):
            return cast(nn.Module, layer.mlp.down_proj)
        else:
            raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected DINOv3ViTGatedMLP or DINOv3ViTMLP.")

    def _get_rc2(self, layer: nn.Module) -> nn.Module:
        return layer

    def _get_pre_hooks_modules(self, layer: nn.Module) -> set[str]:
        r"""
        Returns the names of the modules for which register_forward_pre_hook should be used
        in place of register_forward_hook to access their inputs instead of their outputs.
        """
        if isinstance(layer.mlp, DINOv3ViTGatedMLP):
            return {"rc1", "act"}
        elif isinstance(layer.mlp, DINOv3ViTMLP):
            return {"rc1"}
        else:
            raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected DINOv3ViTGatedMLP or DINOv3ViTMLP.")


# -------------------------------------------------------------------
#   Model
# -------------------------------------------------------------------
class Dinov3(BaseVisionModel):
    r"""
    DINOv3 implementation following [1]_.

    Parameters
    ----------
    config: configuration class with
        model_size: str
            Type of model to use. Options are "vits16", "vitb16", "vitl16", "vit7b16", "vits16plus" and "vith16plus".
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
    .. [1] O. Simeoni et al. DINOv3. arXiv 2025
    """

    def __init__(self, config: Dinov3Config):
        super().__init__()
        self.device = config.device
        self.pooling_type = config.pooling
        self.model_name = f"facebook/dinov3-{config.model_size}-pretrain-lvd1689m"
        self.model = self._load_model(model_name=self.model_name, device=self.device)

    def _load_model(self, model_name: str, device: str) -> PreTrainedModel:
        r"""
        Load weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "facebook/dinov3-vits16-pretrain-lvd1689m",
                "facebook/dinov3-vits16plus-pretrain-lvd1689m",
                "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "facebook/dinov3-vith16plus-pretrain-lvd1689m",
                "facebook/dinov3-vit7b16-pretrain-lvd1689m".
        device: str
            Device to use.

        Notes
        -----
        The model is loaded in evaluation mode by default (dropout disabled). Call `model.train()` explicitly
        if you intend to fine-tune.
        """
        available_models = [
            "facebook/dinov3-vits16-pretrain-lvd1689m",
            "facebook/dinov3-vits16plus-pretrain-lvd1689m",
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")
        model = DINOv3ViTModel.from_pretrained(f"{model_name}", device_map=device)
        model.eval()
        return model

    @property
    def block_adapter(self) -> type[TransformerBlockAdapter]:
        r"""Return the block adapter class for the model's transformer blocks."""
        return Dinov3BlockAdapter

    @property
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        encoder = cast(DINOv3ViTEncoder, self.model.model)
        return cast(ModuleList, encoder.layer)

    @property
    def final_norm(self) -> nn.Module:
        r"""Return the norm applied to the output of the last transformer block."""
        return cast(nn.Module, self.model.norm)

    @property
    def n_prefix_tokens(self) -> int:
        r"""Return the number of auxiliary tokens, i.e., the CLS token followed by the register tokens."""
        return 1 + self.model.config.num_register_tokens
