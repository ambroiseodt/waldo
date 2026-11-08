r"""
Vision encoder of CLIP pretrained on image-text pairs following [1]_. It allows to study specific layers and modules.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension

Notes
-----
We focus on the pretrained models available on HuggingFace for image feature extraction.
See https://huggingface.co/openai for other model variants. Only the vision tower is loaded,
the text tower and the projections of the checkpoint are not used.

References
----------
.. [1] A. Radford, J. W. Kim et al. Learning Transferable Visual Models From Natural Language
       Supervision. In ICML 2021

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
from transformers import CLIPVisionModel, PreTrainedModel
from transformers.models.clip.modeling_clip import CLIPMLP, CLIPEncoder, CLIPVisionTransformer

from ..transformer_block import RC2NORM, TransformerBlockAdapter
from .base import BaseVisionModel

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
        Size of the patches to be extracted from the image. Options are 16 and 32 for "base", and 14 for "large".
    resolution: int
        Image resolution. Options are 224, and 336 for "large".
    pooling: str
        Pooling over the sequence of tokens. Options are "content" (mean over the patch tokens, default),
        "all" (concatenation of the CLS token and the mean over the patch tokens) and "cls" (CLS token only).
    device: str
        Device to use.
    """

    model_size: str = "base"
    patch_size: int = 16
    resolution: int = 224
    pooling: str = "content"
    device: str = "cpu"

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        # Valid model size
        if self.model_size not in ["base", "large"]:
            raise ValueError(f"Invalid model size {self.model_size}. Options are 'base' and 'large'.")

        # Valid patch size
        patch_sizes = {"base": [16, 32], "large": [14]}[self.model_size]
        if self.patch_size not in patch_sizes:
            raise ValueError(
                f"Invalid patch size {self.patch_size} for size '{self.model_size}'. Options are {patch_sizes}."
            )

        # Valid resolution
        resolutions = {"base": [224], "large": [224, 336]}[self.model_size]
        if self.resolution not in resolutions:
            raise ValueError(
                f"Invalid resolution {self.resolution} for size '{self.model_size}'. Options are {resolutions}."
            )

        # Valid pooling
        if self.pooling not in ["content", "all", "cls"]:
            raise ValueError(f"Invalid pooling {self.pooling}. Options are 'content', 'all' and 'cls'.")


# -------------------------------------------------------------------
#   Transformer block adapter
# -------------------------------------------------------------------
class ClipBlockAdapter(TransformerBlockAdapter):
    r"""
    Adapter mapping the standard module names to the submodules of a CLIP transformer block.

    Parameters
    ----------
    layer: nn.Module
        A single CLIP transformer block.

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
        mlp = cast(CLIPMLP, layer.mlp)
        return cast(nn.Module, mlp.fc1)

    def _get_act(self, layer: nn.Module) -> nn.Module:
        mlp = cast(CLIPMLP, layer.mlp)
        return cast(nn.Module, mlp.activation_fn)

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        mlp = cast(CLIPMLP, layer.mlp)
        return cast(nn.Module, mlp.fc2)

    def _get_rc2(self, layer: nn.Module) -> nn.Module:
        return layer

    def _get_pre_hooks_modules(self, layer: nn.Module) -> set[str]:
        r"""
        Returns the names of the modules for which register_forward_pre_hook should be used
        in place of register_forward_hook to access their inputs instead of their outputs.
        """
        return {"rc1"}


# -------------------------------------------------------------------
#   Model
# -------------------------------------------------------------------
class Clip(BaseVisionModel):
    r"""
    Vision encoder of CLIP following [1]_.

    Parameters
    ----------
    config: configuration class with
        model_size: str
            Type of model to use. Options are "base" and "large".
        patch_size: int
            Size of the patches to be extracted from the image. Options are 16 and 32 for "base", and 14 for
            "large".
        resolution: int
            Image resolution. Options are 224, and 336 for "large".
        pooling: str
            Pooling over the sequence of tokens. Options are "content" (mean over the patch tokens, default),
            "all" (concatenation of the CLS token and the mean over the patch tokens) and "cls" (CLS token only).
        device: str
            Device to use.

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled). Call `model.train()` explicitly
    if you intend to fine-tune.

    CLIP applies a layernorm to the embeddings before the first transformer block (pre_layrnorm in HuggingFace),
    which is part of the input of the first block. Its final norm (post_layernorm) is only applied to the CLS
    token, before the projection to the image-text space. As such, rc2norm is always the normed CLS token,
    whatever the pooling type.

    References
    ----------
    .. [1] A. Radford, J. W. Kim et al. Learning Transferable Visual Models From Natural Language
           Supervision. In ICML 2021
    """

    def __init__(self, config: ClipConfig):
        super().__init__()
        self.device = config.device
        self.pooling_type = config.pooling
        suffix = "-336" if config.resolution == 336 else ""
        self.model_name = f"openai/clip-vit-{config.model_size}-patch{config.patch_size}{suffix}"
        self.model = self._load_model(model_name=self.model_name, device=self.device)

    def _load_model(self, model_name: str, device: str) -> PreTrainedModel:
        r"""
        Load the weights of the vision tower from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "openai/clip-vit-base-patch16",
                "openai/clip-vit-base-patch32",
                "openai/clip-vit-large-patch14",
                "openai/clip-vit-large-patch14-336".
        device: str
            Device to use.

        Notes
        -----
        The model is loaded in evaluation mode by default (dropout disabled). Call `model.train()` explicitly
        if you intend to fine-tune.
        """
        available_models = [
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
            "openai/clip-vit-large-patch14-336",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")
        model = CLIPVisionModel.from_pretrained(f"{model_name}", device_map=device)
        model.eval()
        return model

    def get_embeddings(
        self,
        pixel_values: torch.Tensor,
        layers: list[int] | None = None,
        modules: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        r"""
        Single forward pass recovering the pooled sequence of tokens after each specified module in each specified
        transformer block. See BaseVisionModel.get_embeddings.

        RC2NORM is the output of the final norm, which CLIP only applies to the CLS token (pooler_output in
        HuggingFace). It is therefore the normed CLS token whatever the pooling type.

        Raises
        ------
        ValueError
            If RC2NORM is requested without the last layer.
        """
        if modules is None or RC2NORM not in modules:
            return super().get_embeddings(pixel_values=pixel_values, layers=layers, modules=modules)

        # Valid layers, other errors are raised by BaseVisionModel.get_embeddings
        layers = layers if layers is not None else list(range(self.n_layers))
        if self.n_layers - 1 not in [layer % self.n_layers for layer in layers]:
            raise ValueError(f"Module '{RC2NORM}' is only defined for the last layer, which must be requested.")

        # Output of the final norm, of shape (N, E) since it is only applied to the CLS token
        embeddings: dict[str, torch.Tensor] = {}
        key = f"block_{self.n_layers - 1}_{RC2NORM}"

        def hook_fn(module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            embeddings[key] = output.cpu()

        handle = self.final_norm.register_forward_hook(hook_fn)
        try:
            other_modules = [module for module in modules if module != RC2NORM]
            embeddings.update(super().get_embeddings(pixel_values=pixel_values, layers=layers, modules=other_modules))
        finally:
            handle.remove()
        return embeddings

    @property
    def block_adapter(self) -> type[TransformerBlockAdapter]:
        r"""Return the block adapter class for the model's transformer blocks."""
        return ClipBlockAdapter

    @property
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        vision_model = cast(CLIPVisionTransformer, self.model.vision_model)
        encoder = cast(CLIPEncoder, vision_model.encoder)
        return cast(ModuleList, encoder.layers)

    @property
    def final_norm(self) -> nn.Module:
        r"""Return the final norm, which CLIP only applies to the CLS token of the last transformer block."""
        vision_model = cast(CLIPVisionTransformer, self.model.vision_model)
        return cast(nn.Module, vision_model.post_layernorm)

    @property
    def n_prefix_tokens(self) -> int:
        r"""Return the number of auxiliary tokens, i.e., the CLS token."""
        return 1
