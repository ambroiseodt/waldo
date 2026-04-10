r"""
SigLIP 2 implementation following [1]_. It allows to study specific layers and modules.

Since SigLip2 is a vision-language model, we also implement a method to recover the text embedding. This
can be used to study the alignement between image and text embeddings depending on layers and modules.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension

Notes
-----
We focus on the pretrained models available on HuggingFace for zero-shot image classification..
See https://huggingface.co/collections/google/siglip2 for other model variants.

References
----------
.. [1] Tschannen et al. SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding,
       Localization, and Dense Features. arXiv 2025


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
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Siglip2VisionModel,
)
from transformers.models.siglip2.modeling_siglip2 import Siglip2Encoder, Siglip2MLP

from ..transformer_block import TransformerBlockAdapter
from ..utils import mean_pooling
from .base import BaseVisionModel

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
#   Config
# -------------------------------------------------------------------
@dataclass
class Siglip2Config:
    r"""
    Siglip2 configuration file.

    Parameters
    ----------
    model_size: str
        Type of model to use. Options are "base", "large", "giant-opt", and "so400m".
    patch_size: int
        Size of the patches to be extracted from the image. Options are 14, 16 and 32.
    resolution: int
        Image resolution. Options are 224, 256, 384 and 512.
    device: str
        Device to use.
    """

    model_size: str = "base"
    patch_size: int = 16
    resolution: int = 224
    device: str = "cpu"

    def __post_init__(self):
        # Valid model size
        if self.model_size not in ["base", "large", "giant-opt", "so400m"]:
            raise ValueError(
                f"Invalid model size {self.model_size}. Options are 'base', 'large', 'giant-opt' and 'so400m'."
            )

        # Valid patch size
        if self.patch_size not in [14, 16, 32]:
            raise ValueError(f"Invalid patch size {self.patch_size}. Options are 14, 16 and 32.")

        # Valid resolution
        if self.resolution not in [224, 256, 384, 512]:
            raise ValueError(f"Invalid resolution {self.resolution}. Options are 224, 256, 384 and 512.")


# -------------------------------------------------------------------
#   Transformer block adapter
# -------------------------------------------------------------------
class Siglip2BlockAdapter(TransformerBlockAdapter):
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
        ffn = cast(Siglip2MLP, layer.ffn)
        return cast(nn.Module, ffn.fc1)

    def _get_act(self, layer: nn.Module) -> nn.Module:
        ffn = cast(Siglip2MLP, layer.ffn)
        return cast(nn.Module, ffn.activation_fn)

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        ffn = cast(Siglip2MLP, layer.ffn)
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
#   Vision Tower (The Probing Weapon)
# -------------------------------------------------------------------
class Siglip2(BaseVisionModel):
    r"""
    SigLip2 vision model implementation following [1]_.

    The vision encoder of SigLip2 can be studied as a standalone vision model. Since SigLip2 is a
    vision-language model, we also implement a method that can be used to recover the text embeddings
    of the class labels to study the alignement between image and text embeddings.

    Parameters
    ----------
    config: configuration class with
        model_size: str
            Type of model to use. Options are "base", "large", "giant-opt", and "so400m".
        patch_size: int
            Size of the patches to be extracted from the image. Options are 14, 16 and 32.
        resolution: int
            Image resolution. Options are 224, 256, 384 and 512.
        device: str
            Device to use.

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled,
    BatchNorm uses running statistics). Call `model.train()` explicitly
    if you intend to fine-tune.

    References
    ----------
    .. [1] Tschannen et al. SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding,
        Localization, and Dense Features. arXiv 2025
    """

    def __init__(self, config: Siglip2Config):
        super().__init__()
        self.device = config.device
        self.model_name = f"google/siglip2-{config.model_size}-patch{config.patch_size}-{config.resolution}"
        self.model = self._load_from_huggingface(model_name=self.model_name, device=self.device)
        self.tokenizer = self._load_tokenizer(model_name=self.model_name)

    def _load_from_huggingface(self, model_name: str, device: str) -> PreTrainedModel:
        r"""
        Load weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "google/siglip2-base-patch16-224",
                "google/siglip2-base-patch16-256",
                "google/siglip2-base-patch16-384",
                "google/siglip2-base-patch16-512",
                "google/siglip2-base-patch32-224",
                "google/siglip2-large-patch16-224",
                "google/siglip2-large-patch16-256",
                "google/siglip2-large-patch16-384",
                "google/siglip2-large-patch16-512",
                "google/siglip2-giant-opt-patch16-256",
                "google/siglip2-giant-opt-patch16-384",
                "google/siglip2-so400m-patch14-384",
                "google/siglip2-so400m-patch16-256",
                "google/siglip2-so400m-patch16-384",
                "google/siglip2-so400m-patch16-512".
        device: str
            Device to use.

        Notes
        -----
        The model is set to evaluation mode after loading (dropout disabled).
        Call model.train() explicitly to re-enable training mode.
        """
        available_models = [
            "google/siglip2-base-patch16-224",
            "google/siglip2-base-patch16-256",
            "google/siglip2-base-patch16-384",
            "google/siglip2-base-patch16-512",
            "google/siglip2-base-patch32-224",
            "google/siglip2-large-patch16-224",
            "google/siglip2-large-patch16-256",
            "google/siglip2-large-patch16-384",
            "google/siglip2-large-patch16-512",
            "google/siglip2-giant-opt-patch16-256",
            "google/siglip2-giant-opt-patch16-384",
            "google/siglip2-so400m-patch14-384",
            "google/siglip2-so400m-patch16-256",
            "google/siglip2-so400m-patch16-384",
            "google/siglip2-so400m-patch16-512",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")

        # We set ignore_mismatched_sizes=True is required because the official checkpoint keys 'logit_scale' does not
        # align with the HuggingFace implementation. This does not affect the integrity of the pretrained weights.
        model = Siglip2VisionModel.from_pretrained(f"{model_name}", device_map=device, ignore_mismatched_sizes=True)
        model.eval()
        return model

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

        In SigLip2, there is no CLS token, so we do not need to exclude it before the global pooling.

        Parameters
        ----------
        hidden_states: torch.Tensor of shape (N, L, E)
            Sequence of tokens to pool.

        Returns
        -------
        torch.Tensor of shape (N, E)
            Pooled embedding stored on CPU.
        """
        return mean_pooling(hidden_states, exclude_cls=False).cpu()

    @torch.inference_mode()
    def get_class_embeddings_dict(self, text_labels: list[str]) -> torch.Tensor:
        r"""
        Return the embeddings of the text labels of a given dataset.

        Parameters
        ----------
        text_labels: list of length n_classes
            List of text labels of the dataset classes.

        Returns
        -------
        torch.Tensor of shape n_classes x embedding dimension
            The i-th entry is the text embeddings of the i-th class in the dataset.
        """
        # Recover encoding of the text labels
        encoding = self.tokenizer(
            text_labels,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        print(encoding)

        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask

        # Recover text embeddings
        embeddings = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)

        # Recover the dictionnary of text embeddings
        return embeddings

    @property
    def block_adapter(self) -> type[TransformerBlockAdapter]:
        r"""Return the block adapter class for the model's transformer blocks."""
        return Siglip2BlockAdapter

    @property
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        encoder = cast(Siglip2Encoder, self.model.encoder)
        return cast(ModuleList, encoder.layers)
