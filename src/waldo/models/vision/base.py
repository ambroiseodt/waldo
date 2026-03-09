r"""
Abstract model for large vision models (LVMs). It allows to study specific layer and modules following [1, 2]_.

Reference
---------
.. [1] A. Odonnat et al. Vision Transformer Finetuning Benefits from Smooth Components. arXiv 2026

.. [2] A. Odonnat et al. Layer by layer, module by module: Choose both for optimal
       OOD probing of ViT. In ICLR 2026 CAO Workshop

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from abc import ABC, abstractmethod

import torch.nn as nn

# -------------------------------------------------------------------
# Block adapter
# -------------------------------------------------------------------


class TransformerBlockAdapter(ABC):
    r"""
    Adapter that maps standard module names to a transformer block's submodules:
        ln1: first layer norm
        attn:  multi-head attention
        ln2: second layer norm
        ffn_fc1: first feedforward linear layer
        ffn_act: feedforward activation function
        ffn_fc2: second feedforward linear layer

    Parameters
    ----------
    layer: nn.Module
        A single transformer block.

    Usage
    -----
    Create a subclass that implement the six abstract methods for a specific architecture:
    >>> class ViTBlockAdapter(TransformerBlockAdapter):
    ...     def _get_ln1(self, layer):     return layer.layernorm_before
    ...     def _get_attn(self, layer):    return layer.attention
    ...     def _get_ln2(self, layer):     return layer.layernorm_after
    ...     def _get_fc1(self, layer): return layer.intermediate
    ...     def _get_act(self, layer): return layer.intermediate.intermediate_act_fn
    ...     def _get_fc2(self, layer): return layer.output
    """

    def __init__(self, layer: nn.Module):
        self.ln1 = self._get_ln1(layer)
        self.attn = self._get_attn(layer)
        self.ln2 = self._get_ln2(layer)
        self.fc1 = self._get_fc1(layer)
        self.act = self._get_act(layer)
        self.fc2 = self._get_fc2(layer)

    @abstractmethod
    def _get_ln1(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_attn(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_ln2(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_fc1(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_act(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_fc2(self, layer: nn.Module) -> nn.Module: ...

    def module_map(self) -> dict[str, nn.Module]:
        return {
            "ln1": self.ln1,
            "attn": self.attn,
            "ln2": self.ln2,
            "fc1": self.fc1,
            "act": self.act,
            "fc2": self.fc2,
        }


# # base.py — add BaseViT after TransformerBlockAdapter

# from abc import ABC, abstractmethod
# from typing import cast

# import torch
# import torch.nn as nn
# from torch.nn import ModuleList
# from transformers import PreTrainedModel


# class BaseViT(nn.Module, ABC):
#     r"""
#     Abstract base class for Vision Transformer models.

#     Handles the full encoding pipeline in a single forward pass using hooks.
#     Subclasses only need to implement three things:

#         - _load_from_huggingface: load and return the HuggingFace model
#         - block_adapter: return the adapter class for this model's blocks
#         - blocks: return the ModuleList of transformer blocks

#     Parameters
#     ----------
#     device: str
#         Device to run the model on.

#     Notes
#     -----
#     The model is loaded in evaluation mode by default (dropout disabled).
#     Call model.train() explicitly if you intend to fine-tune.

#     Examples
#     --------
#     >>> class ViT(BaseViT):
#     ...     def __init__(self, config: ViTConfig):
#     ...         super().__init__(config.device)
#     ...         self.model_name = f"vit-{config.model_size}-patch{config.patch_size}-{config.resolution}-in21k"
#     ...         self.model = self._load_from_huggingface(self.model_name, self.device)
#     ...
#     ...     def _load_from_huggingface(self, model_name, device):
#     ...         model = ViTModel.from_pretrained(f"google/{model_name}", device_map=device)
#     ...         model.eval()
#     ...         return model
#     ...
#     ...     @property
#     ...     def block_adapter(self): return ViTBlockAdapter
#     ...
#     ...     @property
#     ...     def blocks(self):
#     ...         encoder = cast(ViTEncoder, self.model.encoder)
#     ...         return cast(ModuleList, encoder.layer)
#     """

#     def __init__(self, device: str):
#         super().__init__()
#         self.device = device
#         self.model: PreTrainedModel  # must be set in subclass __init__

#     @abstractmethod
#     def _load_from_huggingface(self, model_name: str, device: str) -> PreTrainedModel:
#         """Load and return the HuggingFace model in evaluation mode on the specified device."""
#         ...

#     @property
#     @abstractmethod
#     def block_adapter(self) -> type[TransformerBlockAdapter]:
#         """Return the adapter class for this model's transformer blocks."""
#         ...

#     @property
#     @abstractmethod
#     def blocks(self) -> ModuleList:
#         """Return the list of transformer blocks."""
#         ...

#     def gap(self, x: torch.Tensor) -> torch.Tensor:
#         r"""
#         Global average pooling over the sequence of patch tokens.

#         Parameters
#         ----------
#         x: torch.Tensor of shape (N, L, E)
#             Embedding sequence to pool.

#         Returns
#         -------
#         emb: torch.Tensor of shape (N, E)
#             Pooled embedding stored on CPU.

#         Notes
#         -----
#         Other pooling strategies could be used, e.g., CLS token extraction.
#         """
#         return x.mean(dim=1).cpu()

#     @torch.inference_mode()
#     def get_embeddings(
#         self,
#         x: torch.Tensor,
#         layers: list[int] | None = None,
#         modules: list[str] | None = None,
#     ) -> dict[str, torch.Tensor]:
#         r"""
#         Single forward pass recovering token embeddings after each specified
#         module in each specified transformer block.

#         Parameters
#         ----------
#         x: torch.Tensor of shape (N, C, H, W)
#             Preprocessed input image on the same device as the model.
#         layers: list[int] | None, default=None
#             Layer indices to recover. If None, all layers are recovered.
#         modules: list[str] | None, default=None
#             Module names from ALL_MODULES. If None, all modules are recovered.

#         Returns
#         -------
#         embeddings: dict[str, torch.Tensor]
#             Maps "block_{layer_idx}_{module_name}" to a tensor of shape (N, E)
#             stored on CPU.
#         """
#         layers = layers if layers is not None else list(range(self.n_layers))
#         modules = modules if modules is not None else ALL_MODULES

#         embeddings: dict[str, torch.Tensor] = {}
#         handles: list[torch.utils.hooks.RemovableHook] = []

#         def make_hook(key: str):
#             def hook_fn(
#                 module: nn.Module,
#                 input: tuple[torch.Tensor, ...],
#                 output: torch.Tensor | tuple[torch.Tensor, ...],
#             ) -> None:
#                 out = output[0] if isinstance(output, tuple) else output
#                 embeddings[key] = self.gap(out)

#             return hook_fn

#         layer_set = set(layers)
#         for layer_idx, layer in enumerate(self.blocks):
#             if layer_idx not in layer_set:
#                 continue
#             adapter = self.block_adapter(layer)
#             for module_name, module in adapter.module_map().items():
#                 if module_name not in modules:
#                     continue
#                 key = f"block_{layer_idx}_{module_name}"
#                 handles.append(module.register_forward_hook(make_hook(key)))

#         try:
#             need_hidden = "res1" in modules or "res2" in modules
#             outputs = self.model(x, output_hidden_states=need_hidden)
#         finally:
#             # Always remove hooks after the forward pass, even if an exception
#             # is raised. Otherwise hooks accumulate on the model and corrupt
#             # future embeddings.
#             for handle in handles:
#                 handle.remove()

#         if need_hidden:
#             for layer_idx in layers:
#                 if "res1" in modules:
#                     embeddings[f"block_{layer_idx}_res1"] = self.gap(outputs.hidden_states[layer_idx])
#                 if "res2" in modules:
#                     embeddings[f"block_{layer_idx}_res2"] = self.gap(outputs.hidden_states[layer_idx + 1])

#         return embeddings

#     def __repr__(self) -> str:
#         n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
#         return f"{self.__class__.__name__} — {n_params:,} trainable parameters."

#     @property
#     def embedding_dim(self) -> int:
#         return self.model.config.hidden_size

#     @property
#     def n_layers(self) -> int:
#         return self.model.config.num_hidden_layers
