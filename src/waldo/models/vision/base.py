r"""
Abstract model for vision foundation models (VFMs). It allows to study specific layers and modules following [1, 2]_.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension


References
----------
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

from ..pooling import mean_pooling
from ..transformer_block import RC2NORM, TRANSFORMER_MODULES, TransformerBlockAdapter


# -------------------------------------------------------------------
# Base Vision Model
# -------------------------------------------------------------------
class BaseVisionModel(nn.Module, ABC):
    r"""
    Abstract base class for transformer-based vision foundation models.

    It handles the full encoding pipeline in a single forward pass using hooks and pre-hooks.
    Subclasses only need to implement:
        - _load_model: load and return the HuggingFace model
        - block_adapter: returns the adapter class for this model's blocks
        - blocks: returns the ModuleList of transformer blocks
        - final_norm: returns the norm applied to the output of the last transformer block
        - n_prefix_tokens: returns the number of auxiliary tokens (CLS and registers) excluded from the pooling
    and set pooling_type from their configuration ("content" by default, see pooling).

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
    ...     @property
    ...     def block_adapter(self): return ViTBlockAdapter
    ...
    ...     @property
    ...     def blocks(self):
    ...         encoder = cast(ViTEncoder, self.model.encoder)
    ...         return cast(ModuleList, encoder.layer)
    ...
    ...     @property
    ...     def final_norm(self): return self.model.layernorm
    ...
    ...     @property
    ...     def n_prefix_tokens(self): return 1
    """

    def __init__(self):
        super().__init__()
        self.device: str
        self.model: PreTrainedModel
        self.pooling_type: str = "content"

    @abstractmethod
    def _load_model(self, model_name: str, device: str) -> PreTrainedModel:
        r"""Load a pretrained HuggingFace model in evaluation mode on the specified device."""
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
    def final_norm(self) -> nn.Module:
        r"""
        Return the norm applied to the output of the last transformer block, whose output is the standard output
        of the model (last_hidden_state in HuggingFace). It is used to probe the rc2norm module.
        """
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

    def pooling(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Pooling over the sequence of tokens to obtain a global embedding stored on CPU. Since the embeddings are
        obtained from get_embeddings that uses @torch.inference_mode(), there is no need to detach the tensor.

        Two pooling types are available, set by pooling_type:
            - "content": mean pooling over the patch tokens, i.e., excluding the n_prefix_tokens auxiliary tokens
              (CLS and registers). This is the default.
            - "all": concatenation of the CLS token and the mean pooling over the patch tokens, following the linear
              probing protocol of DINOv2 and DINOv3 [1, 2]_. Register tokens are excluded. The embedding dimension
              is doubled. It requires a CLS token.

        Parameters
        ----------
        hidden_states: torch.Tensor of shape (N, L, E)
            Sequence of tokens to pool.

        Returns
        -------
        torch.Tensor of shape (N, E) or (N, 2E)
            Global sequence representation after pooling stored on CPU.

        References
        ----------
        .. [1] M. Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. In TMLR 2024

        .. [2] O. Simeoni et al. DINOv3. arXiv 2025
        """
        embedding = mean_pooling(hidden_states=hidden_states, n_prefix_tokens=self.n_prefix_tokens)
        if self.pooling_type == "all":
            embedding = torch.cat([hidden_states[:, 0], embedding], dim=-1)
        return embedding.cpu()

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
            Layer indices to recover. Negative indices follow the Python convention, e.g., -1 is the last layer.
            If None, all layers are recovered.
        modules: list[str] | None, default=None
            Module names from TRANSFORMER_MODULES, and RC2NORM for the output of the final norm (only defined for
            the last layer, which must then be among the layers). If None, all modules from TRANSFORMER_MODULES
            are recovered.

        Returns
        -------
        embeddings: dict[str, torch.Tensor]
            Maps "block_{layer_idx}_{module_name}" to the pooled patch embeddings of shape (N, E) stored on CPU,
            with nonnegative layer indices.

        Raises
        ------
        ValueError
            If a layer index is out of range, a module name is invalid, or RC2NORM is requested without the last layer.
        """
        # Recover layers and modules
        layers = layers if layers is not None else list(range(self.n_layers))
        modules = modules if modules is not None else TRANSFORMER_MODULES

        # Valid layers and modules
        invalid_layers = [layer for layer in layers if not -self.n_layers <= layer < self.n_layers]
        if invalid_layers:
            raise ValueError(
                f"Layer indices {invalid_layers} are out of range for a model with {self.n_layers} layers."
            )
        valid_modules = TRANSFORMER_MODULES + [RC2NORM]
        invalid_modules = [module for module in modules if module not in valid_modules]
        if invalid_modules:
            raise ValueError(f"Invalid modules {invalid_modules}. Options are {valid_modules}.")
        layers = [layer % self.n_layers for layer in layers]
        if RC2NORM in modules and self.n_layers - 1 not in layers:
            raise ValueError(f"Module '{RC2NORM}' is only defined for the last layer, which must be requested.")

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

        # Output of the final norm, only defined for the last layer
        if RC2NORM in modules:
            key = f"block_{self.n_layers - 1}_{RC2NORM}"
            handles.append(self.final_norm.register_forward_hook(self._make_hook(embeddings, key)))

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
