r"""
Abstract model for vision foundation models (VFMs). It allows to study specific layers and modules following [1, 2]_.

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

from ..transformer_block import TRANSFORMER_MODULES, TransformerBlockAdapter


# -------------------------------------------------------------------
# Base Vision Model
# -------------------------------------------------------------------
class BaseVisionModel(nn.Module, ABC):
    r"""
    Abstract base class for transformer-based vision foundation models.

    It handles the full encoding pipeline in a single forward pass using hooks and pre-hooks.
    Subclasses only need to implement:
        - _load_model: load and return the HuggingFace model
        - pooling: returns the pooling method over sequences of tokens
        - block_adapter: returns the adapter class for this model's blocks
        - blocks: returns the ModuleList of transformer blocks
        - n_prefix_tokens: returns the number of auxiliary tokens (CLS and registers) excluded from the pooling

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled). Call model.train() explicitly if you intend
    to fine-tune. To add a new method, we advise users to follow this order for clarity:
        1. __init__
        2. abstract private methods
        3. abstract public methods
        4. abstract properties
        5. private methods
        6. public methods
        7. dunder methods
        8. properties

    Usage
    -----
    >>> class ViT(BaseVisionModel):
    ...     def __init__(self, config: ViTConfig):
    ...         super().__init__()
    ...         self.device = config.device
    ...         self.model_name = f"google/vit-{config.model_size}-patch{config.patch_size}-{config.resolution}-in21k"
    ...         self.model = self._load_model(model_name=self.model_name, device=self.device)
    ...
    ...     def _load_model(self, model_name: str, device: str):
    ...         model = ViTModel.from_pretrained(f"{model_name}", device_map=device)
    ...         model.eval()
    ...         return model
    ...
    ...     def pooling(self, hidden_states: torch.Tensor):
    ...         return mean_pooling(hidden_states=hidden_states, n_prefix_tokens=self.n_prefix_tokens).cpu()
    ...
    ...     @property
    ...     def block_adapter(self): return ViTBlockAdapter
    ...
    ...     @property
    ...     def blocks(self):
    ...         encoder = cast(ViTEncoder, self.model.encoder)
    ...         return cast(ModuleList, encoder.layer)
    ...
    ...     @property
    ...     def n_prefix_tokens(self): return 1
    """

    def __init__(self):
        super().__init__()
        self.device: str
        self.model: PreTrainedModel

    @abstractmethod
    def _load_model(self, model_name: str, device: str) -> PreTrainedModel:
        r"""Load a pretrained HuggingFace model in evaluation mode on the specified device."""
        ...

    @abstractmethod
    def pooling(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Pooling method over the sequence of tokens to obtain a global embedding.

        Parameters
        ----------
        hidden_states: torch.Tensor of shape (N, L, E)
            Sequence of tokens to pool.

        Returns
        -------
        torch.Tensor of shape (N, E)
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

    @property
    @abstractmethod
    def n_prefix_tokens(self) -> int:
        r"""
        Return the number of auxiliary tokens at the start of the sequence, i.e., the CLS token and the
        register tokens if any. They are excluded from the pooling, which is done over patch tokens only.
        """
        ...

    def _make_pre_hook(self, embeddings: dict[str, torch.Tensor], key: str) -> Callable:
        r"""
        Create a forward pre-hook that captures the INPUT of a module.
        The pooling is done at the hook level for efficiency.

        Parameters
        ----------
        embeddings: dict[str, torch.Tensor]
            Shared dict where the captured embedding will be stored.
        key: str
            Key under which the embedding is stored, e.g. "block_3_rc1".

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
            embeddings[key] = self.pooling(hidden_states=input[0])

        return pre_hook_fn

    def _make_hook(
        self,
        embeddings: dict[str, torch.Tensor],
        key: str,
        processor: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> Callable:
        r"""
        Create a forward hook that captures the OUTPUT of a module.
        An optional processor can transform the output before pooling,
        for instance to recover FC1 module in DINOv2.

        Parameters
        ----------
        embeddings: dict[str, torch.Tensor]
            Shared dict where the captured embedding will be stored.
        key: str
            Key under which the embedding is stored, e.g. "block_3_mha".
        processor: Callable[[torch.Tensor], torch.Tensor] | None
            Optional function applied to the module output before pooling.

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

            if processor is not None:
                out = processor(out)

            # Recover embedding after global average pooling over the sequence of tokens
            embeddings[key] = self.pooling(hidden_states=out)

        return hook_fn

    @torch.inference_mode()
    def get_embeddings(
        self,
        pixel_values: torch.Tensor,
        layers: list[int] | None = None,
        modules: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        r"""
        Single forward pass recovering the pooled sequence of tokens of shape (N, E) after each specified
        module in each specified transformer block.

        Parameters
        ----------
        pixel_values: torch.Tensor of shape (N, C, H, W)
            Preprocessed input image on the same device as the model.
        layers: list[int] | None, default=None
            Nonnegative layer indices to recover. If None, all layers are recovered.
        modules: list[str] | None, default=None
            Module names from TRANSFORMER_MODULES. If None, all modules are recovered.

        Returns
        -------
        embeddings: dict[str, torch.Tensor]
            Maps "block_{layer_idx}_{module_name}" to the pooled patch embeddings of shape (N, E) stored on CPU.

        Raises
        ------
        ValueError
            If a layer index is out of range or a module name is not in TRANSFORMER_MODULES.
        """
        # Recover layers and modules
        layers = layers if layers is not None else list(range(self.n_layers))
        modules = modules if modules is not None else TRANSFORMER_MODULES

        # Valid layers and modules
        invalid_layers = [layer for layer in layers if not 0 <= layer < self.n_layers]
        if invalid_layers:
            raise ValueError(
                f"Layer indices {invalid_layers} are out of range for a model with {self.n_layers} layers."
            )
        invalid_modules = [module for module in modules if module not in TRANSFORMER_MODULES]
        if invalid_modules:
            raise ValueError(f"Invalid modules {invalid_modules}. Options are {TRANSFORMER_MODULES}.")

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
                    handles.append(
                        module.register_forward_hook(
                            self._make_hook(
                                embeddings,
                                key,
                                processor=lambda out, adapter=adapter, name=module_name: adapter.process_output(
                                    name, out
                                ),
                            )
                        )
                    )

        # Forward pass
        try:
            self.model(pixel_values=pixel_values)

        # Always remove hooks after the forward pass, even if an exception
        # is raised. Otherwise hooks accumulate on the model and corrupt
        # future embeddings.
        finally:
            for handle in handles:
                handle.remove()
        return embeddings

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return f"{self.__class__.__name__} model with {n_params:,} trainable parameters."

    @property
    def embedding_dim(self) -> int:
        r"""Return the embedding dimension."""
        return self.model.config.hidden_size

    @property
    def n_layers(self) -> int:
        r"""Return the number of transformer blocks."""
        return len(self.blocks)
