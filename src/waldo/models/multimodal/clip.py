r"""
CLIP implementation following [1]_. It allows to study specific layers and modules.

Since CLIP is a vision-language model, we also implement a method to recover the text embedding. This
can be used to study the alignement between image and text embeddings depending on layers and modules.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension

Notes
-----
We focus on the pretrained models available on HuggingFace for zero-shot image classification..
See https://huggingface.co/openai/models for other model variants.

References
----------
.. [1] Radford et al. Learning Transferable Visual Models From Natural Language Supervision. arXiv 2021


See also
--------
TransformerBlockAdapter
BaseVisionModel

@ 2026, Ambroise Odonnat
"""

import logging
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch.nn import ModuleList
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPTextModel,
    CLIPVisionModel,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.models.clip.modeling_clip import CLIPMLP, CLIPEncoder, CLIPVisionTransformer
from transformers.processing_utils import ProcessorMixin

from ..pooling import mean_pooling
from ..transformer_block import TransformerBlockAdapter
from .base import BaseVisionLanguageModel

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
#   Config
# -------------------------------------------------------------------
@dataclass
class ClipConfig:
    r"""
    CLIP configuration file.

    Parameters
    ----------
    model_size: str
        Type of model to use. Options are "base" and "large".
    patch_size: int
        Size of the patches to be extracted from the image. Options are 14, 16 and 32.
    device: str
        Device to use.
    """

    model_size: str = "base"
    patch_size: int = 16
    device: str = "cpu"

    def __post_init__(self):
        # Valid model size
        if self.model_size not in ["base", "large"]:
            raise ValueError(f"Invalid model size {self.model_size}. Options are 'base' and 'large'.")

        # Valid patch size
        if self.patch_size not in [14, 16, 32]:
            raise ValueError(f"Invalid patch size {self.patch_size}. Options are 14, 16 and 32..")


# -------------------------------------------------------------------
#   Transformer block adapter
# -------------------------------------------------------------------
class ClipBlockAdapter(TransformerBlockAdapter):
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
        return cast(nn.Module, layer.layer_norm1)

    def _get_mha(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.self_attn)

    def _get_rc1(self, layer: nn.Module) -> nn.Module:
        # Use pre_hook to capture rc1
        return cast(nn.Module, layer.layer_norm2)

    def _get_ln2(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.layer_norm2)

    def _get_fc1(self, layer: nn.Module) -> nn.Module:
        ffn = cast(CLIPMLP, layer.mlp)
        return cast(nn.Module, ffn.fc1)

    def _get_act(self, layer: nn.Module) -> nn.Module:
        ffn = cast(CLIPMLP, layer.mlp)
        return cast(nn.Module, ffn.activation_fn)

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        ffn = cast(CLIPMLP, layer.mlp)
        return cast(nn.Module, ffn.fc2)

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
class Clip(BaseVisionLanguageModel):
    r"""
    CLIP vision model implementation following [1]_.

    The vision encoder of CLIP can be studied as a standalone vision model. Since CLIP is a
    vision-language model, we also implement a method that can be used to recover the text embeddings
    of the class labels to study the alignement between image and text embeddings.

    Parameters
    ----------
    config: configuration class with
        model_size: str
            Type of model to use. Options are "base" and "large".
        patch_size: int
            Size of the patches to be extracted from the image. Options are 14, 16 and 32.
        device: str
            Device to use.

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled,
    BatchNorm uses running statistics). Call `model.train()` explicitly
    if you intend to fine-tune.

    References
    ----------
    .. [1] Radford et al. Learning Transferable Visual Models From Natural Language Supervision. arXiv 2021
    """

    def __init__(self, config: ClipConfig):
        super().__init__()
        self.device = config.device
        self.model_name = f"openai/clip-vit-{config.model_size}-patch{config.patch_size}"
        self.vision_model = self._load_vision_model(model_name=self.model_name, device=self.device)
        self.text_model = self._load_text_model(model_name=self.model_name, device=self.device)
        self.processor = self._load_processor(model_name=self.model_name)
        self.tokenizer = self._load_tokenizer(model_name=self.model_name)

    def _load_vision_model(self, model_name: str, device: str) -> PreTrainedModel:
        r"""
        Load vision weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "openai/clip-vit-base-patch16",
                "openai/clip-vit-base-patch32",
                "openai/clip-vit-large-patch14".
        device: str
            Device to use.

        Notes
        -----
        The model is loaded in evaluation mode by default (dropout disabled,
        BatchNorm uses running statistics). Call `model.train()` explicitly
        if you intend to fine-tune.

        """
        available_models = [
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")

        model = CLIPVisionModel.from_pretrained(f"{model_name}", device_map=device)
        model.eval()
        return model

    def _load_text_model(self, model_name: str, device: str) -> PreTrainedModel:
        r"""
        Load text weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "openai/clip-vit-base-patch16",
                "openai/clip-vit-base-patch32",
                "openai/clip-vit-large-patch14".
        device: str
            Device to use.

        Notes
        -----
        The model is loaded in evaluation mode by default (dropout disabled,
        BatchNorm uses running statistics). Call `model.train()` explicitly
        if you intend to fine-tune.
        """
        available_models = [
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")
        model = CLIPTextModel.from_pretrained(f"{model_name}", device_map=device)
        model.eval()
        return model

    def _load_processor(self, model_name: str) -> ProcessorMixin:
        r"""
        Load the image processor from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model.
        """
        return AutoProcessor.from_pretrained(model_name)

    def _load_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        r"""
        Load the tokenizer from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model.
        """
        return AutoTokenizer.from_pretrained(model_name)

    def pooling(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Mean pooling over patch tokens stored on cpu. Since the embeddings are obtained
        from get_embeddings that uses @torch_inference, there is no need to detach the tensor.

        In CLIP, there is a CLS token, so we need to exclude it before the global pooling.

        Parameters
        ----------
        hidden_states: torch.Tensor of shape (N, L, E)
            Sequence of tokens to pool.

        Returns
        -------
        torch.Tensor of shape (N, E)
            Pooled embedding stored on CPU.
        """
        return mean_pooling(hidden_states, exclude_cls=True).cpu()

    @property
    def block_adapter(self) -> type[TransformerBlockAdapter]:
        r"""Return the block adapter class for the model's transformer blocks."""
        return ClipBlockAdapter

    @property
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        vision_model = cast(CLIPVisionTransformer, self.vision_model.vision_model)
        encoder = cast(CLIPEncoder, vision_model.encoder)
        return cast(ModuleList, encoder.layers)
