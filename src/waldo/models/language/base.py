r"""
Abstract base class for large language models (LLMs). It allows to study specific layer and modules following [1, 2]_.

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

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..transformer_block import TRANSFORMER_MODULES, TransformerBlockAdapter

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Base Language Model
# -------------------------------------------------------------------
class BaseLanguageModel(nn.Module, ABC):
    r"""
    Abstract base class for transformer-based large language models.

    It handles tokenization, sentence mask construction, hook registration, and mean pooling in a single forward pass.
    The encode() method is compatible with the MTEB v2 EncoderProtocol: it accepts a DataLoader[BatchedInput] and
    returns mean-pooled embeddings from the specified layer and module, excluding padding, BOS, and EOS tokens.
    Subclasses only need to implement:
        - _load_from_huggingface: load and return the HuggingFace model
        - _load_tokenizer: load and return the tokenizer
        - block_adapter: return the adapter class for this model's blocks
        - blocks: return the ModuleList of transformer blocks

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled). Call model.train() explicitly if you intend
    to fine-tune. Unlike BaseVisionModel, hooks capture the full (N, L, E) tensor rather than pooling inside the hook.
    This is necessary to apply the sentence mask after the forward pass, since the mask depends on input_ids which are
    only available at encode() time. To add a new method, we advise users to follow this order for clarity:
        1. __init__
        2. abstract private methods
        3. abstract public methods
        4. abstract properties
        5. private methods
        6. public methods
        7. dunder methods
        8. properties
    """

    def __init__(self):
        super().__init__()
        self.device: str
        self.model: PreTrainedModel
        self.tokenizer: PreTrainedTokenizerBase
        self.layer: int | None = None
        self.module: str | None = None

    @abstractmethod
    def _load_from_huggingface(self, model_name: str, dtype: torch.dtype, device: str):
        r"""Load a pretrained HuggingFace model in evaluation mode on the specified device."""
        ...

    @abstractmethod
    def _load_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        r"""Load the tokenizer for the model."""
        ...

    @abstractmethod
    def pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        special_token_ids: list[int | None] | None = None,
    ) -> torch.Tensor:
        r"""
        Pooling method over the sequence of tokens to obtain a global embedding.

        Parameters
        ----------
        hidden_states: torch.Tensor of shape (N, L, E)
            Sequence of tokens to pool.
        attention_mask: torch.Tensor of shape (N, L) | None, default=None
            Padding mask (1 for real tokens, 0 for padding). If None, no padding tokens are excluded.
        input_ids: torch.Tensor of shape (N, L) | None, default=None
            Integer tensor mapping each token to its vocabulary index, as returned by the tokenizer.
            Used to identify and exclude special tokens by id. Required when special_token_ids is set.
        special_token_ids: list[int | None] | None, default=None
            Token ids to exclude from pooling, e.g. [bos_token_id, eos_token_id]. None values are ignored
            to handle tokenizers that miss some special tokens. When defined, input_ids must be provided.

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
    def special_token_ids(self) -> list[int | None]:
        r"""Return the list of special tokens to excluse from the pooling."""
        ...

    def _make_pre_hook(self, embeddings: dict[str, torch.Tensor], key: str) -> Callable:
        r"""
        Create a forward pre-hook that captures the full INPUT tensor of a module.
        Unlike BaseVisionModel, no pooling is applied inside the hook: the full
        (N, L, E) tensor is stored for later mean pooling with the sentence mask.

        Parameters
        ----------
        embeddings: dict[str, torch.Tensor]
            Shared dict where the captured tensor will be stored.
        key: str
            Key under which the tensor is stored, e.g. "block_3_rc1".

        Returns
        -------
        pre_hook_fn: Callable
            A function compatible with register_forward_pre_hook.
        """

        def pre_hook_fn(
            module: nn.Module,
            input: tuple[torch.Tensor, ...],
        ) -> None:
            embeddings[key] = input[0]

        return pre_hook_fn

    def _make_hook(self, embeddings: dict[str, torch.Tensor], key: str) -> Callable:
        r"""
        Create a forward hook that captures the full OUTPUT tensor of a module.
        Unlike BaseVisionModel, no pooling is applied inside the hook: the full
        (N, L, E) tensor is stored for later mean pooling with the sentence mask.

        Parameters
        ----------
        embeddings: dict[str, torch.Tensor]
            Shared dict where the captured tensor will be stored.
        key: str
            Key under which the tensor is stored, e.g. "block_3_mha".

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
            out = output[0] if isinstance(output, tuple) else output
            embeddings[key] = out

        return hook_fn

    @torch.inference_mode()
    def _get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Single forward pass recovering the sequence of tokens of shape (N, L, E) after the specified layer and module.

        Parameters
        ----------
        input_ids: torch.Tensor of shape (N, L)
            Tokenized input on the same device as the model.
        attention_mask: torch.Tensor of shape (N, L)
            Padding mask on the same device as the model.
        layer: int
            Layer index to recover.
        module: str
            Module name from TRANSFORMER_MODULES to recover.

        Returns
        -------
        hidden_states: torch.Tensor of shape (N, L, E)
            Token embeddings before pooling.
        """
        embeddings: dict[str, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        # Instantiate layer and module
        if self.layer is None or self.module is None:
            raise RuntimeError(
                "Layer and module must be set before encoding. Call model.set_probe(layer=..., module=...) first."
            )
        module = self.module
        layer = self.layer
        key = f"block_{layer}_{module}"
        adapter = self.block_adapter(self.blocks[layer])
        module_obj = adapter.module_map()[module]

        if module in adapter.pre_hooks:
            handles.append(module_obj.register_forward_pre_hook(self._make_pre_hook(embeddings=embeddings, key=key)))
        else:
            handles.append(module_obj.register_forward_hook(self._make_hook(embeddings=embeddings, key=key)))

        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for handle in handles:
                handle.remove()

        # Recover hidden states
        hidden_states = embeddings[key]

        return hidden_states

    def _encode_sentences(
        self,
        sentences: list[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        r"""
        Encode a flat list of sentences into mean-pooled embeddings.

        Parameters
        ----------
        sentences: list[str]
            List of input sentences or documents to encode.
        batch_size: int, default=32
            Number of sentences to process per forward pass.

        Returns
        -------
        embeddings: torch.Tensor of shape (N, E)
            Sentence embeddings stored on CPU.
        """
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]

            # Recover encoding, input_ids and padding tokens (attention_mask)
            encoding = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            input_ids = encoding.input_ids
            attention_mask = encoding.attention_mask

            # Recover hidden states
            hidden_states = self._get_hidden_states(input_ids=input_ids, attention_mask=attention_mask)

            # Recover pooled embeddings
            embeddings = self.pooling(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                input_ids=input_ids,
                special_token_ids=self.special_token_ids,
            )
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def set_probe(self, layer: int, module: str) -> None:
        r"""
        Set the layer and module to probe for embedding extraction.

        Allows reusing the same model instance across multiple (layer, module) pairs without reloading weights.

        Parameters
        ----------
        layer: int
            Layer index to probe. Negative indices are supported (-1 = last layer).
        module: str
            Module name from TRANSFORMER_MODULES to probe.
        """
        # Set layer
        if layer > self.n_layers or layer >= self.n_layers:
            logger.warning(
                f"Layer index {layer} is out of range for a model with {self.n_layers} layers. "
                f"Defaulting to the last layer (index=-1)."
            )
            layer = -1
        self.layer = layer

        # Set module
        if module not in TRANSFORMER_MODULES:
            raise ValueError(f"Invalid module '{module}'. Options are {TRANSFORMER_MODULES}.")
        self.module = module

    def encode(
        self,
        inputs: DataLoader,
        *,
        task_metadata: Any = None,
        hf_split: str = "",
        hf_subset: str = "",
        prompt_type: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Encode inputs compatible with MTEB v2 EncoderProtocol.

        Unpacks the DataLoader[BatchedInput] into a flat list of sentences,
        then encodes them in batches using _encode_sentences.

        Parameters
        ----------
        inputs: DataLoader[BatchedInput]
            Batched inputs from MTEB containing text under the "text" key.
        task_metadata: TaskMetadata | None
            Metadata of the current MTEB task.
        hf_split: str
            Current dataset split.
        hf_subset: str
            Current dataset subset.
        prompt_type: PromptType | None
            Query or passage prompt type.

        Returns
        -------
        embeddings: torch.Tensor of shape (N, E)
            Sentence embeddings stored on CPU.
        """
        # Unpack DataLoader[BatchedInput] to flat list of strings
        sentences = [text for batch in inputs for text in batch["text"]]
        return self._encode_sentences(sentences)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return f"{self.__class__.__name__} model with {n_params:,} trainable parameters."

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers
