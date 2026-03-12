r"""
Abstract model for large vision models (LVMs). It allows to study specific layer and modules following [1, 2]_.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension


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
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.nn import ModuleList
from transformers import PreTrainedModel

TRANSFORMER_MODULES = ["ln1", "mha", "rc1", "ln2", "fc1", "act", "fc2", "rc2"]

# -------------------------------------------------------------------
# Block adapter
# -------------------------------------------------------------------


class TransformerBlockAdapter(ABC):
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
    Our implementation assumes that pre-norm is used, as is the case in modern transformer models like ViT, DINO,
    or Llama, i.e., normalization layers are applied before the attention and feedforward blocks (see [1, 2]). It
    allows us to recover all modules using a single forward pass by using register_forward_hook, except for rc1,
    which can be seen as the input to ln2 and is recovered using register_forward_pre_hook.  We advice users that
    would like to use a post-norm standard to make sure their implementation matches our design. For generality, the
    modules that require a pre_hook can be specified in the _pre_hook_modules method. This is notably needed for
    modern models that rely on a gated activation like SwiGLU in the FFN (see [3, 4]_ for details). For consistency,
    we decompose the FFN block as follows:

        fc1  = gate_proj(x)                      -> linear layer which feeds the activation
        act  = silu(gate_proj(x)) * up_proj(x)   -> non-linearity (which is a function of the input too)
        fc2  = down_proj(act)                    -> linear layer which gives the output of the FFN

    This ensures that fc1 and fc2 remain pure linear operations, and the gating mechanism is absorbed into act. The
    up_proj can be interpreted as part of the activation definition rather than a separate linear layer, making act
    the complete sequence-to-sequence non-linearity between fc1 and fc2. Subclasses must reflect this convention in
    their _get_fc1, _get_act, and _get_fc2 implementations, and declare act as a pre-hook module in _pre_hook_modules
    when SwiGLU is used, since the gated activation output is recovered as the input to down_proj.


    References
    ----------
    .. [1] R. Child et al. Generating Long Sequences with Sparse Transformers. arXiv 2019

    .. [2] R. Xiong et al. On Layer Normalization in the Transformer Architecture. In ICML 2020

    .. [3] Y. Dauphin et al. Language Modeling with Gated Convolutional Networks. In ICML 2017

    .. [4] N. Shazeer. GLU Variants Improve Transformer. arXiv 2020


    Usage
    -----
    >>> class ViTBlockAdapter(TransformerBlockAdapter):
    ...     def _get_ln1(self, layer):  return layer.layernorm_before
    ...     def _get_mha(self, layer):  return layer.attention
    ...     def _get_rc1(self, layer):  return layer.layer_norm_after  # pre_hook captures rc1
    ...     def _get_ln2(self, layer):  return layer.layernorm_after
    ...     def _get_fc1(self, layer):  return layer.intermediate.dense
    ...     def _get_act(self, layer):  return layer.intermediate.intermediate_act_fn
    ...     def _get_fc2(self, layer):  return layer.output.dense
    ...     def _get_rc2(self, layer):  return layer.output
    """

    def __init__(self, layer: nn.Module):
        self.ln1 = self._get_ln1(layer)
        self.mha = self._get_mha(layer)
        self.rc1 = self._get_rc1(layer)
        self.ln2 = self._get_ln2(layer)
        self.fc1 = self._get_fc1(layer)
        self.act = self._get_act(layer)
        self.fc2 = self._get_fc2(layer)
        self.rc2 = self._get_rc2(layer)
        self.pre_hooks = self._get_pre_hooks_modules(layer)

    @abstractmethod
    def _get_ln1(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_mha(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_rc1(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_ln2(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_fc1(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_act(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_fc2(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_rc2(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_pre_hooks_modules(self, layer: nn.Module) -> set[str]:
        r"""
        Returns the names of the module for which register_forward_pre_hook should be used
        in place of register_forward_hook to access their inputs instead of the outputs.

        """
        ...

    def module_map(self) -> dict[str, nn.Module]:
        return {
            "ln1": self.ln1,
            "mha": self.mha,
            "rc1": self.rc1,
            "ln2": self.ln2,
            "fc1": self.fc1,
            "act": self.act,
            "fc2": self.fc2,
            "rc2": self.rc2,
        }


# -------------------------------------------------------------------
# Base Vision Model
# -------------------------------------------------------------------
class BaseVisionModel(nn.Module, ABC):
    r"""
    Abstract base class for transformer-based vision foundation models.

    It handles the full encoding pipeline in a single forward pass using hooks and pre-hooks.
    Subclasses only need to implement three things:
        - _load_from_huggingface: load and return the HuggingFace model
        - pooling: returns the pooling method over sequences of tokens
        - block_adapter: returns the adapter class for this model's blocks
        - blocks: returns the ModuleList of transformer blocks

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled). Call model.train() explicitly if you intend
    to fine-tune. To add a new method, we advise users to follow this order for clarity and consistency:
        1. __init__
        2. abstract private methods
        3. abstract public methods
        4. abstract properties
        5. private methods
        6. public methods
        7. dunder methods
        8. properties

    Usage
    --------
    >>> class ViT(BaseVisionModel):
    ...     def __init__(self, config: ViTConfig):
    ...         super().__init__()
    ...         self.device = config.device
    ...         self.model_name = f"google/vit-{config.model_size}-patch{config.patch_size}-{config.resolution}-in21k"
    ...         self.model = self._load_from_huggingface(model_name=self.model_name, device=self.device)
    ...
    ...     def _load_from_huggingface(self, model_name: str, device: str):
    ...         model = ViTModel.from_pretrained(f"{model_name}", device_map=device)
    ...         model.eval()
    ...         return model
    ...
    ...     def pooling(self, x: torch.Tensor): return mean_pooling(x=x, exclude_cls=True).cpu()
    ...
    ...     @property
    ...     def block_adapter(self): return ViTBlockAdapter
    ...
    ...     @property
    ...     def blocks(self):
    ...         encoder = cast(ViTEncoder, self.model.encoder)
    ...         return cast(ModuleList, encoder.layer)
    """

    def __init__(self):
        super().__init__()
        self.device: str
        self.model: PreTrainedModel

    @abstractmethod
    def _load_from_huggingface(self, model_name: str, device: str) -> PreTrainedModel:
        r"""Load a pretrained HuggingFace model in evaluation mode on the specified device."""
        ...

    @abstractmethod
    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Pooling method over the sequence of tokens to obtain a global embedding.

        Parameters
        ----------
        x: torch.Tensor of shape (N, L, E)
            Sequence of tokens to pool.

        Returns
        -------
        x_pooled: torch.Tensor of shape (N, E)
            Global sequence representation after pooling stored on CPU.
        """
        ...

    @property
    @abstractmethod
    def block_adapter(self) -> type[TransformerBlockAdapter]:
        r"""Return the block adapter class for the model's transformer blocks."""
        ...

    @property
    @abstractmethod
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        ...

    def _make_pre_hook(self, embeddings: dict[str, torch.Tensor], key: str) -> Callable:
        r"""
        Create a forward pre-hook that captures the INPUT of a module.

        Parameters
        ----------
        embeddings: dict[str, torch.Tensor]
            Shared dict where the captured embedding will be stored.
        key: str
            Key under which the embedding is stored, e.g. "block_3_res1".

        Returns
        -------
        pre_hook_fn: Callable
            A function compatible with register_forward_pre_hook.
        """

        def pre_hook_fn(
            module: nn.Module,
            input: tuple[torch.Tensor, ...],
        ) -> None:

            # Recover embedding after global average pooling over the sequence of tokens
            embeddings[key] = self.pooling(input[0])

        return pre_hook_fn

    def _make_hook(self, embeddings: dict[str, torch.Tensor], key: str) -> Callable:
        r"""
        Create a forward hook that captures the OUTPUT of a module.

        Parameters
        ----------
        embeddings: dict[str, torch.Tensor]
            Shared dict where the captured embedding will be stored.
        key: str
            Key under which the embedding is stored, e.g. "block_3_mha".

        Returns
        -------
        hook_fn: Callable
            A function compatible with register_forward_hook.
        """

        def hook_fn(
            module: nn.Module,
            input: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            # Some modules output a tuple in HuggingFace, e.g. the attention module
            out = output[0] if isinstance(output, tuple) else output

            # Recover embedding after global average pooling over the sequence of tokens
            embeddings[key] = self.pooling(out)

        return hook_fn

    @torch.inference_mode()
    def get_embeddings(
        self,
        x: torch.Tensor,
        layers: list[int] | None = None,
        modules: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        r"""
        Single forward pass recovering token embeddings after each specified
        module in each specified transformer block.

        Parameters
        ----------
        x: torch.Tensor of shape (N, C, H, W)
            Preprocessed input image on the same device as the model.
        layers: list[int] | None, default=None
            Layer indices to recover. If None, all layers are recovered.
        modules: list[str] | None, default=None
            Module names from TRANSFORMER_MODULES. If None, all modules are recovered.

        Returns
        -------
        embeddings: dict[str, torch.Tensor]
            Maps "block_{layer_idx}_{module_name}" to a tensor of shape (N, E) stored on CPU.
        """
        # Recover layers and modules
        layers = layers if layers is not None else list(range(self.n_layers))
        modules = modules if modules is not None else TRANSFORMER_MODULES

        # Initialize embeddings and handles
        embeddings: dict[str, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        # Loop over transformer blocks to register hooks across specified layers and modules
        layer_set = set(layers)
        for layer_idx, layer in enumerate(self.blocks):
            if layer_idx not in layer_set:
                continue
            adapter = self.block_adapter(layer)
            for module_name, module in adapter.module_map().items():
                if module_name not in modules:
                    continue
                key = f"block_{layer_idx}_{module_name}"

                # Modules for which we hook the inputs
                if module_name in adapter.pre_hooks:
                    handles.append(
                        module.register_forward_pre_hook(self._make_pre_hook(embeddings=embeddings, key=key))
                    )
                # Modules for which we hook the output
                else:
                    handles.append(module.register_forward_hook(self._make_hook(embeddings=embeddings, key=key)))

        # Forward pass
        try:
            self.model(x)

        # Always remove hooks after the forward pass, even if an exception
        # is raised. Otherwise hooks accumulate on the model and corrupt
        # future embeddings.
        finally:
            for handle in handles:
                handle.remove()

        return embeddings

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return f"{self.__class__.__name__} — {n_params:,} trainable parameters."

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers
