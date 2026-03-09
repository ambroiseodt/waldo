r"""
Vision Transformer model pretrained on ImageNet-21k following [1]_.

Comments abbreviations:
    N: batch size
    L: sequence length
    E: embedding dimension

References
----------
.. [1] A. Dosovitskiy, L. Beyer et al. An Image is Worth 16x16 Words: Transformers
       for Image Recognition at Scale. ICLR 2021.

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
from transformers import PreTrainedModel, ViTModel
from transformers.models.vit.modeling_vit import ViTEncoder, ViTIntermediate

from .base import TransformerBlockAdapter

logger = logging.getLogger(__name__)

ALL_MODULES = ["ln1", "attn", "res1", "ln2", "fc1", "act", "fc2", "res2"]


# -------------------------------------------------------------------
# ViT configuration
# -------------------------------------------------------------------
@dataclass
class ViTConfig:
    r"""
    ViT configuration file.

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
# ViT block adapter
# -------------------------------------------------------------------


class ViTBlockAdapter(TransformerBlockAdapter):
    def _get_ln1(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.layernorm_before)

    def _get_attn(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.attention)

    def _get_ln2(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.layernorm_after)

    def _get_fc1(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.intermediate)

    def _get_act(self, layer: nn.Module) -> nn.Module:
        intermediate = cast(ViTIntermediate, layer.intermediate)
        return cast(nn.Module, intermediate.intermediate_act_fn)

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.output)


# -------------------------------------------------------------------
# ViT model
# -------------------------------------------------------------------
class ViT(nn.Module):
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

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.model_name = f"vit-{config.model_size.lower()}-patch{config.patch_size}-{config.resolution}-in21k"
        self.device = config.device
        self.model = self._load_from_huggingface(model_name=self.model_name, device=self.device)

    def _load_from_huggingface(self, model_name: str, device: str) -> PreTrainedModel:
        r"""
        Load weights into the model from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "vit-base-patch16-224-in21k",
                "vit-base-patch32-224-in21k",
                "vit-large-patch16-224-in21k",
                "vit-large-patch32-224-in21k",
                "vit-huge-patch14-224-in21k".
        device: str
            Device to use.

        Notes
        -----
        The model is loaded in evaluation mode by default (dropout disabled,
        BatchNorm uses running statistics). Call `model.train()` explicitly
        if you intend to fine-tune.
        """
        # Check model_name validity
        available_models = [
            "vit-base-patch16-224-in21k",
            "vit-base-patch32-224-in21k",
            "vit-large-patch16-224-in21k",
            "vit-large-patch32-224-in21k",
            "vit-huge-patch14-224-in21k",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")

        # Load pretrained weights
        model = ViTModel.from_pretrained(f"google/{model_name}", device_map=device)
        model.eval()

        return model

    def gap(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Apply global average pooling on patch tokens. To allow for a fair comparison across layers, modules and between
        models, the CLS token is not taken into account and we focus on the information contained in patches.

        Parameters
        ----------
        x: torch.Tensor of dimension (N, L, E)
            Embedding sequence to pool.

        Returns
        -------
        emb: torch.Tensor of dimension (N, E)
            Output after pooling.

        Notes
        -----
        Other type of pooling method could be designed.
        """
        emb = x.mean(dim=1).cpu()
        return emb

    @torch.inference_mode()
    def get_embeddings(
        self,
        x: torch.Tensor,
        layers: list[int] | None = None,
        modules: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        r"""
        Single forward pass allowing to recover the sequence of tokens obtained
        after each specified module in each specified transformer block.

        Parameters
        ----------
        x: torch.Tensor of shape (N, C, H, W)
            Input image after preprocessing. It should be on the same device as the model.
        layers: list[int] | None, default=None
            Indices of layers to recover. If None, all layers are recovered.
        modules: list[str] | None, default=None
            Name of modules to recover from ALL_MODULES. If None, all
            modules are recovered.

        Returns
        -------
        embeddings: dict[str, torch.Tensor]
            Dictionnary containg the embeddings across layers and modules. The entry to the key
            "{layer_idx}_{module_name}" is a tensor of shape (N, E) stored on CPU.
        """

        # Recover layers and modules
        layers = layers if layers is not None else list(range(self.n_layers))
        modules = modules if modules is not None else ALL_MODULES

        # Initialize embeddings and handles
        embeddings = {}
        handles = []

        def make_hook(key: str):
            def hook_fn(
                module: nn.Module,
                input: tuple[torch.Tensor, ...],
                output: torch.Tensor | tuple[torch.Tensor, ...],
            ) -> None:

                # Some module might output a tuple in HuggingFace, e.g., the attention module
                out = output[0] if isinstance(output, tuple) else output

                # Recover embedding after global average pooling over the sequence of tokens
                embeddings[key] = self.gap(out)

            return hook_fn

        # Loop over transformer blocks to register hooks across specified layers and modules
        layer_set = set(layers)
        for layer_idx, layer in enumerate(self.blocks):
            if layer_idx not in layer_set:
                continue
            adapter = ViTBlockAdapter(layer)
            for module_name, module in adapter.module_map().items():
                if module_name not in modules:
                    continue
                key = f"block_{layer_idx}_{module_name}"
                handles.append(module.register_forward_hook(make_hook(key)))

        # Single forward pass
        try:
            residual_connections = "res1" in modules or "res2" in modules
            outputs = self.model(x, output_hidden_states=residual_connections)

        # Always remove hooks after the forward pass, even if an exception is raised.
        # Otherwise, hooks would accumulate on the model and fire on every subsequent
        # forward pass, corrupting future embeddings.
        finally:
            for handle in handles:
                handle.remove()

        # Recover residual connections
        if "res1" in modules or "res2" in modules:
            for layer_idx in layers:
                if "res1" in modules:
                    embeddings[f"block_{layer_idx}_res1"] = self.gap(outputs.hidden_states[layer_idx])
                if "res2" in modules:
                    embeddings[f"block_{layer_idx}_res2"] = self.gap(outputs.hidden_states[layer_idx + 1])

        return embeddings

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return f"Model with {n_params} trainable parameters."

    @property
    def blocks(self) -> ModuleList:
        encoder = cast(ViTEncoder, self.model.encoder)
        return cast(ModuleList, encoder.layer)

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers
