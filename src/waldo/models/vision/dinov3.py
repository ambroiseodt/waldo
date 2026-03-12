r"""
DINOv3 ViT-based pretrained on LVD-1689m following [1]_. It allows to study specific layer and modules.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension

References
----------
.. [1] O. Simeoni et al. DINOv3. arXiv 2025

See also
--------
TransformerBlockAdapter and BaseVisionModel in base.py

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
from transformers import DINOv3ViTModel, PreTrainedModel
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTGatedMLP, DINOv3ViTMLP

from ..utils import mean_pooling
from .base import BaseVisionModel, TransformerBlockAdapter

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Dinov3 configuration
# -------------------------------------------------------------------
@dataclass
class Dinov3Config:
    r"""
    Dinov2 configuration file.

    Parameters
    ----------
    model_size: str
        Type of model to use. Options are "vits16", "vitb16", "vitl16", "vit7b16", "vits16plus" and "vith16plus".
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
        if self.model_size not in ["vits16", "vitb16", "vitl16", "vit7b16", "vits16plus", "vith16plus"]:
            raise ValueError(
                f"Invalid model size {self.model_size}."
                " Options are 'vits16', 'vitb16', 'vitl16', 'vitl16', 'vit7b16', 'vits16plus'and 'vith16plus'."
            )


# -------------------------------------------------------------------
# Dinov3 block adapter
# -------------------------------------------------------------------


class Dinov3BlockAdapter(TransformerBlockAdapter):
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
    DINOv3-7B model relies on a gated SwiGLU activation in the FFN (see [3, 4]_ for details). For consistency
    with ViT, we decompose the FFN block as follows:

        fc1  = gate_proj(x)                      -> linear layer which feeds the activation
        act  = silu(gate_proj(x)) * up_proj(x)   -> non-linearity (which is a function of the input too)
        fc2  = down_proj(act)                    -> linear layer which gives the output of the FFN

    This ensures that fc1 and fc2 remain pure linear operations, and the gating mechanism is absorbed into act. The
    up_proj can be interpreted as part of the activation definition rather than a separate linear layer, making act
    the complete sequence-to-sequence non-linearity between fc1 and fc2. Subclasses must reflect this convention in
    their _get_fc1, _get_act, and _get_fc2 implementations, and declare act as a pre-hook module in _pre_hook_modules
    when SwiGLU is used, since the gated activation output is recovered as the input to down_proj. Note that Dinov2
    makes use of slight modifications of the transformer architecture (see, e.g., LayerScale [1]_) that do not
    fundamentally change the transformer architecture and merely improve memory-efficiency and training
    stability. For the consistency of our analysis, we do not probe them.

    References
    ----------
    .. [1] H. Touvron et al. Going Deeper With Image Transformers. ICCV 2021

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
            return cast(nn.Module, layer.mlp.gated_proj)
        if isinstance(layer.mlp, DINOv3ViTMLP):
            return cast(nn.Module, layer.mlp.up_proj)
        raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected DINOv3ViTGatedMLP or DINOv3ViTMLP.")

    def _get_act(self, layer: nn.Module) -> nn.Module:
        if isinstance(layer.mlp, DINOv3ViTGatedMLP):
            # Use pre_hook to capture act
            return cast(nn.Module, layer.mlp.down_proj)
        if isinstance(layer.mlp, DINOv3ViTMLP):
            return cast(nn.Module, layer.mlp.act_fn)
        raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected DINOv3ViTGatedMLP or DINOv3ViTMLP.")

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        if isinstance(layer.mlp, (DINOv3ViTGatedMLP, DINOv3ViTMLP)):
            return cast(nn.Module, layer.mlp.down_proj)
        raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected DINOv3ViTGatedMLP or DINOv3ViTMLP.")

    def _get_rc2(self, layer: nn.Module) -> nn.Module:
        return layer

    def _get_pre_hooks_modules(self, layer: nn.Module) -> set[str]:
        r"""
        Returns the names of the module for which register_forward_pre_hook should be used
        in place of register_forward_hook to access their inputs instead of the outputs.
        """
        if isinstance(layer.mlp, DINOv3ViTGatedMLP):
            return {"rc1", "act"}
        if isinstance(layer.mlp, DINOv3ViTMLP):
            return {"rc1"}
        raise ValueError(f"Unsupported FFN type: {type(layer.mlp)}. Expected DINOv3ViTGatedMLP or DINOv3ViTMLP.")


# -------------------------------------------------------------------
# Dinov3 model
# -------------------------------------------------------------------
class Dinov3(BaseVisionModel):
    r"""
    Vision Transformer implementation following [1]_.


    Parameters
    ----------
    config: configuration class with
        model_size: str
            Type of model to use. Options are "vits16", "vitb16", "vitl16", "vit7b16", "vits16plus" and "vith16plus".

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled,
    BatchNorm uses running statistics). Call `model.train()` explicitly
    if you intend to fine-tune.

    References
    ----------
    .. [1] M. Oquab et al. DINOv2: Learning Robust Visual Feature without Supervision. In TMLR 2024
    """

    def __init__(self, config: Dinov3Config):
        super().__init__()
        self.device = config.device
        self.model_name = f"facebook/dinov3-{config.model_size.lower()}-pretrain-lvd1689m"
        self.model = self._load_from_huggingface(model_name=self.model_name, device=self.device)

    def _load_from_huggingface(self, model_name: str, device: str) -> PreTrainedModel:
        r"""
        Load weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "facebook/dinov3-vits16-pretrain-lvd1689m",
                "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "facebook/dinov3-vit7b16-pretrain-lvd1689m",
                "facebook/dinov3-vits16plus-pretrain-lvd1689m",
                "facebook/dinov3-vith16plus-pretrain-lvd1689m".
        device: str
            Device to use.

        Notes
        -----
        The model is set to evaluation mode after loading (dropout disabled).
        Call model.train() explicitly to re-enable training mode.
        """
        available_models = [
            "facebook/dinov3-vits16-pretrain-lvd1689m",
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "facebook/dinov3-vit7b16-pretrain-lvd1689m",
            "facebook/dinov3-vits16plus-pretrain-lvd1689m",
            "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")
        model = DINOv3ViTModel.from_pretrained(f"{model_name}", device_map=device)
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
        return Dinov3BlockAdapter

    @property
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        return cast(ModuleList, self.model.layer)
