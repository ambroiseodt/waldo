r"""
Abstract model for vision-language models (VLMs). It allows to study specific layers and modules following [1, 2]_.

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
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.processing_utils import ProcessorMixin

from ..transformer_block import TRANSFORMER_MODULES, TransformerBlockAdapter


# -------------------------------------------------------------------
# Base Vision-Language Model
# -------------------------------------------------------------------
class BaseVisionLanguageModel(nn.Module, ABC):
    r"""
    Abstract base class for transformer-based vision-language models. The component and layer-wise study is done
    on the vision part while the language part is used to study the alignement between modalities.

    It handles the full encoding pipeline in a single forward pass using hooks and pre-hooks.
    Subclasses only need to implement three things:
        - _load_vision_model: load and return the HuggingFace vision model
        - _load_text_model: load and return the HuggingFace text model
        - _load_processor: load and return the image processor
        - _load_tokenizer: load and return the tokenizer
        - pooling: returns the pooling method over sequences of tokens
        - block_adapter: returns the adapter class for this model's blocks
        - blocks: returns the ModuleList of transformer blocks

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
    >>> class Clip(BaseVisionLanguageModel):
    ...     def __init__(self, config: ClipConfig):
    ...         super().__init__()
    ...         self.device = config.device
    ...         self.model_name = f"openai/clip-vit-{config.model_size}-patch{config.patch_size}"
    ...         self.vision_model = self._load_vision_model(model_name=self.model_name, device=self.device)
    ...         self.text_model = self._load_text_model(model_name=self.model_name, device=self.device)
    ...         self.processor = self._load_processor(model_name=self.model_name)
    ...         self.tokenizer = self._load_tokenizer(model_name=self.model_name)
    ...
    ...     def _load_vision_model(self, model_name: str, dtype: torch.dtype, device: str):
    ...         model = CLIPVisionModel.from_pretrained(f"{model_name}", dtype=dtype, device_map=device)
    ...         model.eval()
    ...         return model
    ...
    ...     def _load_text_model(self, model_name: str, dtype: torch.dtype, device: str):
    ...         model = CLIPTextModel.from_pretrained(f"{model_name}", dtype=dtype, device_map=device)
    ...         model.eval()
    ...         return model
    ...
    ...     def _load_processor(self, model_name: str):
    ...         return Autoprocessor.from_pretrained(model_name)
    ...
    ...     def _load_tokenizer(self, model_name: str):
    ...         return AutoTokenizer.from_pretrained(model_name)
    ...
    ...     def pooling(self, x: torch.Tensor): return mean_pooling(x=x, exclude_cls=True).cpu()
    ...
    ...     @property
    ...     def block_adapter(self): return ClipBlockAdapter
    ...
    ...     @property
    ...     def blocks(self):
    ...         vision_model = cast(CLIPVisionTransformer, self.vision_model.vision_model)
    ...         encoder = cast(CLIPEncoder, vision_model.encoder)
    ...         return cast(ModuleList, encoder.layers)
    """

    def __init__(self):
        super().__init__()
        self.device: str
        self.vision_model: PreTrainedModel
        self.text_model: PreTrainedModel
        self.processor: ProcessorMixin
        self.tokenizer: PreTrainedTokenizerBase

    @abstractmethod
    def _load_vision_model(self, model_name: str, device: str) -> PreTrainedModel:
        r"""Load a pretrained HuggingFace vision model in evaluation mode on the specified device."""
        ...

    @abstractmethod
    def _load_text_model(self, model_name: str, device: str) -> PreTrainedModel:
        r"""Load a pretrained HuggingFace text model in evaluation mode on the specified device."""
        ...

    @abstractmethod
    def _load_processor(self, model_name: str) -> ProcessorMixin:
        r"""Load the image processor from the HuggingFace Transformers library."""
        ...

    @abstractmethod
    def _load_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        r"""Load the tokenizer from the HuggingFace Transformers library."""
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

    def _make_pre_hook(self, embeddings: dict[str, torch.Tensor], key: str) -> Callable:
        r"""
        Create a forward pre-hook that captures the INPUT of a module.
        The pooling is done at the hood level for efficiency.

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

    def _make_hook(self, embeddings: dict[str, torch.Tensor], key: str) -> Callable:
        r"""
        Create a forward hook that captures the OUTPUT of a module.
        The pooling is done at the hood level for efficiency.

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
            Layer indices to recover. If None, all layers are recovered.
        modules: list[str] | None, default=None
            Module names from TRANSFORMER_MODULES. If None, all modules are recovered.

        Returns
        -------
        embeddings: dict[str, torch.Tensor]
            Maps "block_{layer_idx}_{module_name}" to the pooled patch embeddings of shape (N, E) stored on CPU.

        Notes
        -----
        Modern models like SigLIP2 require 'patchification' and spatial metadata before the forward pass.
        To support those implementations, we use the image processor associated with the vision-language model.
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
            # Input pixel values are already processed so the processor only has to patchify the input images
            processor_kwargs = {"do_resize": False, "do_rescale": False, "do_normalize": False, "return_tensors": "pt"}
            encoding = self.processor(images=pixel_values, **processor_kwargs)
            self.vision_model(**encoding)

        # Always remove hooks after the forward pass, even if an exception
        # is raised. Otherwise hooks accumulate on the model and corrupt
        # future embeddings.
        finally:
            for handle in handles:
                handle.remove()

        return embeddings

    @torch.inference_mode()
    def get_class_embeddings(self, text_labels: list[str], text_length: int = 64) -> torch.Tensor:
        r"""
        Return the embeddings of the text labels of a given dataset.

        Parameters
        ----------
        text_labels: list of length n_classes
            List of text labels of the dataset classes.
        text_length: int, default=64
            Maximal sequence length allowed.

        Returns
        -------
        torch.Tensor of shape n_classes x embedding dimension
            The i-th entry is the text embeddings of the text label of the i-th class in the dataset.
        """
        # Recover encoding of the text labels
        encoding = self.tokenizer(
            text=text_labels,
            padding="max_length",
            max_length=text_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Recover text embeddings
        embeddings = self.text_model(**encoding).pooler_output

        # Recover the dictionnary of text embeddings
        return embeddings

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.vision_model.parameters() if p.requires_grad)
        return f"{self.__class__.__name__} model with {n_params:,} trainable parameters."

    @property
    def embedding_dim(self) -> int:
        r"""Return the embedding dimension."""
        return self.vision_model.config.hidden_size

    @property
    def n_layers(self) -> int:
        r"""Return the number of transformer blocks."""
        return len(self.blocks)
