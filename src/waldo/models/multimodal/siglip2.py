r"""
SigLIP2 implementation following [1]_. It allows to study specific layers and modules.

Since SigLIP2 is a vision-language model, we also implement a method to recover the text embedding. This
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
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Siglip2TextModel,
    Siglip2VisionModel,
    SiglipTextModel,
    SiglipVisionModel,
)
from transformers.models.siglip2.modeling_siglip2 import (
    Siglip2Encoder,
    Siglip2MLP,
    Siglip2VisionTransformer,
)
from transformers.processing_utils import ProcessorMixin

from ..pooling import mean_pooling
from ..transformer_block import TransformerBlockAdapter
from .base import BaseVisionLanguageModel

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
#   Config
# -------------------------------------------------------------------
@dataclass
class Siglip2Config:
    r"""
    SigLIP2 configuration file.

    Parameters
    ----------
    model_size: str
        Type of model to use. Options are "base", "large", "giant-opt", and "so400m".
    patch_size: int
        Size of the patches to be extracted from the image. Options are 14, 16, and 32.
    resolution: int | str
        Image resolution. Options are 224, 256, 384, 512 and "naflex".
    device: str
        Device to use.
    """

    model_size: str = "base"
    patch_size: int = 16
    resolution: int | str = 224
    device: str = "cpu"

    def __post_init__(self):
        # Valid model size
        if self.model_size not in ["base", "large", "giant-opt", "so400m"]:
            raise ValueError(
                f"Invalid model size {self.model_size}. Options are 'base', 'large', 'giant-opt' and 'so400m'."
            )

        # Valid patch size
        if self.patch_size not in [14, 16, 32]:
            raise ValueError(f"Invalid patch size {self.patch_size}. Options are 14, 16, and 32.")

        # Valid resolution
        if self.resolution not in [224, 256, 384, 512, "naflex"]:
            raise ValueError(f"Invalid resolution {self.resolution}. Options are 224, 256, 384, 512 and 'naflex'.")


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
        ffn = cast(Siglip2MLP, layer.mlp)
        return cast(nn.Module, ffn.fc1)

    def _get_act(self, layer: nn.Module) -> nn.Module:
        ffn = cast(Siglip2MLP, layer.mlp)
        return cast(nn.Module, ffn.activation_fn)

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        ffn = cast(Siglip2MLP, layer.mlp)
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
class Siglip2(BaseVisionLanguageModel):
    r"""
    SigLIP2 vision model implementation following [1]_.

    The vision encoder of SigLip2 can be studied as a standalone vision model. Since SigLip2 is a
    vision-language model, we also implement a method that can be used to recover the text embeddings
    of the class labels to study the alignement between image and text embeddings.

    Parameters
    ----------
    config: configuration class with
        model_size: str
            Type of model to use. Options are "base", "large", "giant-opt", and "so400m".
        patch_size: int
            Size of the patches to be extracted from the image. Options are 14, 16, and 32.
        resolution: int | str
            Image resolution. Options are 224, 256, 384, 512 and "naflex".
        device: str
            Device to use.

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled, BatchNorm uses running statistics).
    Call `model.train()` if you intend to fine-tune. SigLIP2 allows a NaFlex implementation that supports native
    resolution (by opposition to FixRes variants). In particular, NaFlex variants need already patched images
    (3D tensors with the batch dimension) as input to the forward require 3D patch sequences while the FixRes
    ones require 4D tensors (with the batch dimension). Since the default HuggingFace implementation follows the NaFlex
    variant, it throws errors with FixRes checkpoints. We fix this using SiglipVisionModel and SiglipTextModel to load
    FixRes checkpoints. This works because SigLIP2 has the same architecture as SigLip (except for the image embedding).
    This is not a smart fix but allows for an efficient loading. Future transformers versions might solve this.

    References
    ----------
    .. [1] Tschannen et al. SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding,
        Localization, and Dense Features. arXiv 2025
    """

    def __init__(self, config: Siglip2Config):
        super().__init__()
        self.device = config.device
        self.model_name = f"google/siglip2-{config.model_size}-patch{config.patch_size}-{config.resolution}"
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
                "google/siglip2-base-patch16-224",
                "google/siglip2-base-patch16-256",
                "google/siglip2-base-patch16-384",
                "google/siglip2-base-patch16-512",
                "google/siglip2-base-patch16-naflex",
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
                "google/siglip2-so400m-patch16-naflex".
        device: str
            Device to use.

        Notes
        -----
        The model is loaded in evaluation mode by default (dropout disabled, BatchNorm uses running statistics).
        Call `model.train()` if you intend to fine-tune. SigLIP2 allows a NaFlex implementation that supports native
        resolution (by opposition to FixRes variants). In particular, NaFlex variants need already patched images
        (3D tensors with the batch dimension) as input to the forward require 3D patch sequences while the FixRes
        ones require 4D tensors (with the batch dimension). Since the default HuggingFace implementation follows the
        NaFlex variant, it throws errors with FixRes checkpoints. We fix this using Siglip models to load the FixRes
        checkpoints. This works because SigLIP2 has the same architecture as SigLip (except for the image embedding).
        This is not a smart fix but allows for an efficient loading. Future transformers versions might solve this.

        """
        available_models = [
            "google/siglip2-base-patch16-224",
            "google/siglip2-base-patch16-256",
            "google/siglip2-base-patch16-384",
            "google/siglip2-base-patch16-512",
            "google/siglip2-base-patch16-naflex",
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
            "google/siglip2-so400m-patch16-naflex",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")

        # Native resolution
        huggingface_model = Siglip2VisionModel if "naflex" in model_name else SiglipVisionModel
        model = huggingface_model.from_pretrained(f"{model_name}", device_map=device)
        model.eval()
        return model

    def _load_text_model(self, model_name: str, device: str) -> PreTrainedModel:
        r"""
        Load text weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "google/siglip2-base-patch16-224",
                "google/siglip2-base-patch16-256",
                "google/siglip2-base-patch16-384",
                "google/siglip2-base-patch16-512",
                "google/siglip2-base-patch16-naflex",
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
                "google/siglip2-so400m-patch16-naflex".
        device: str
            Device to use.

        Notes
        -----
        The model is loaded in evaluation mode by default (dropout disabled, BatchNorm uses running statistics).
        Call `model.train()` if you intend to fine-tune. SigLip2 allows a NaFlex implementation that supports native
        resolution (by opposition to FixRes variants). In particular, NaFlex variants need already patched images
        (3D tensors with the batch dimension) as input to the forward require 3D patch sequences while the FixRes
        ones require 4D tensors (with the batch dimension). Since the default HuggingFace implementation follows the
        NaFlex variant, it throws errors with FixRes checkpoints. We fix this using Siglip models to load the FixRes
        checkpoints. This works because SigLip2 has the same architecture as SigLip (except for the image embedding).
        This is not a smart fix but allows for an efficient loading. Future transformers versions might solve this.
        """
        available_models = [
            "google/siglip2-base-patch16-224",
            "google/siglip2-base-patch16-256",
            "google/siglip2-base-patch16-384",
            "google/siglip2-base-patch16-512",
            "google/siglip2-base-patch16-naflex",
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
            "google/siglip2-so400m-patch16-naflex",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")
        huggingface_model = Siglip2TextModel if "naflex" in model_name else SiglipTextModel
        model = huggingface_model.from_pretrained(f"{model_name}", device_map=device)
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

        In SigLIP2, there is no CLS token, so we do not need to exclude it before the global pooling.

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

    @property
    def block_adapter(self) -> type[TransformerBlockAdapter]:
        r"""Return the block adapter class for the model's transformer blocks."""
        return Siglip2BlockAdapter

    @property
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        vision_model = cast(Siglip2VisionTransformer, self.vision_model.vision_model)
        encoder = cast(Siglip2Encoder, vision_model.encoder)
        return cast(ModuleList, encoder.layers)
