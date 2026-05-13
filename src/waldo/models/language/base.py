r"""
Abstract base class for large language models (LLMs). It allows to study specific layers and modules following [1, 2]_.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension

MTEB Compatibility [3, 4]_
--------------------------
BaseLanguageModel inherits from EncoderProtocol, which is the MTEB v2 abstract contract for encoder models.
This makes any subclass (e.g., Gemma3) directly compatible with mteb.evaluate() without any additional
wrapping, and supports the following MTEB task types out of the box:

    - Classification
    - Clustering
    - Semantic Textual Similarity (STS)
    - Reranking (bi-encoder)
    - Bitext Mining
    - Retrieval (MTEB automatically wraps EncoderProtocol into SearchProtocol)

EncoderProtocol requires three concrete methods which are implemented here at the base class level
so that all subclasses inherit them for free:
    - encode: produces mean-pooled sentence embeddings from a specified layer and module.
    - similarity: pairwise cosine similarity between two sets of paired embeddings.
    - similarity_pairwise: full cosine similarity matrix between two sets of embeddings.

Note that CrossEncoderProtocol is a separate MTEB contract for cross-encoder reranking models,
where query and document are encoded jointly rather than independently. This is a fundamentally
different architecture and is not supported here. If cross-encoder reranking with Gemma3 is
needed in the future, a separate BaseLanguageCrossEncoder class should be created inheriting
from CrossEncoderProtocol.

Reference
---------
.. [1] A. Odonnat et al. Vision Transformer Finetuning Benefits from Smooth Components. arXiv 2026

.. [2] A. Odonnat et al. Layer by layer, module by module: Choose both for optimal
       OOD probing of ViT. In ICLR 2026 CAO Workshop

.. [3] N. Muennighoff et al. MTEB: Massive Text Embedding Benchmark. In EACL 2023

.. [4] K. Enevoldsen et al. MMTEB: : Massive Multilingual Text Embedding Benchmark. In ICLR 2025

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Unpack

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta
from mteb.models.models_protocols import EncoderProtocol
from mteb.types import (
    Array,
    BatchedInput,
    EncodeKwargs,
    PromptType,
)
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..transformer_block import TRANSFORMER_MODULES, TransformerBlockAdapter

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Base Language Model
# -------------------------------------------------------------------
class BaseLanguageModel(nn.Module, EncoderProtocol, ABC):
    r"""
    Abstract base class for transformer-based large language models.

    It handles tokenization, sentence mask construction, hook registration, and mean pooling in a single forward pass.
    The encode() method is compatible with the MTEB v2 EncoderProtocol: it accepts a DataLoader[BatchedInput] and
    returns mean-pooled embeddings from the specified layer and module, excluding padding, BOS, and EOS tokens.
    Subclasses only need to implement:
        - _load_model: load and return the HuggingFace model
        - _load_tokenizer: load and return the tokenizer
        - pooling: returns the pooling method over sequences of tokens
        - block_adapter: returns the adapter class for this model's blocks
        - blocks: returns the ModuleList of transformer blocks
        - special_token_ids: returns the list of special token ids to exclude from pooling
        - mteb_model_meta: returns the model metadata needed for compatibility with the MTEB benchmark

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

    Usage
    -----
    >>> class Gemma3(BaseLanguageModel):
    ...     def __init__(self, config: Gemma3Config):
    ...         super().__init__()
    ...         self.device = config.device
    ...         self.model_name = f"google/gemma-3-{config.model_size}-pt"
    ...         self.model = self._load_model(model_name=self.model_name, dtype=self.dtype, device=self.device)
    ...         self.tokenizer = self._load_tokenizer(model_name=self.model_name)
    ...
    ...     def _load_model(self, model_name: str, dtype: torch.dtype, device: str):
    ...         model = Gemma3TextModel.from_pretrained(f"{model_name}", dtype=dtype, device_map=device)
    ...         model.eval()
    ...         return model
    ...
    ...     def _load_tokenizer(self, model_name: str):
    ...         return AutoTokenizer.from_pretrained(model_name)
    ...
    ...     def pooling(
    ...         self,
    ...         hidden_states: torch.Tensor,
    ...         attention_mask: torch.Tensor | None = None,
    ...         input_ids: torch.Tensor | None = None,
    ...         special_token_ids: list[int | None] | None = None,
    ...    ) : return mean_pooling(hidden_states=hidden_states,
    ...                             attention_mask=attention_mask,
    ...                             input_ids=input_ids,
    ...                             special_token_ids=special_token_ids,
    ...                             ).cpu()
    ...
    ...     @property
    ...     def block_adapter(self): return Gemma3BlockAdapter
    ...
    ...     @property
    ...     def blocks(self):
    ...         cast(ModuleList, self.model.layers)
    ...
    ...     @property
    ...     def special_token_ids(self):
    ...         return [
    ...             self.tokenizer.bos_token_id,
    ...             self.tokenizer.eos_token_id,
    ...             self.tokenizer.pad_token_id,
    ...         ]
    ...
    ...     @property
    ...     def mteb_model_meta(self):
    ...         layer = self.layer + self.n_layers if self.layer < 0 else self.layer
    ...         return ModelMeta(
    ...             name=f"{self.model_name}_layer_{layer}_module_{self.module}"),
    ...             revision="main",
    ...             release_date="2025-03-25",
    ...             open_weights=True,
    ...             framework=["PyTorch"],
    ...             license="gemma",
    ...            )
    """

    def __init__(self):
        super().__init__()
        self.device: str
        self.model: PreTrainedModel
        self.tokenizer: PreTrainedTokenizerBase

        # Default probing done on the final output
        self.layer: int = -1
        self.module: str = "rc2"

    @abstractmethod
    def _load_model(self, model_name: str, dtype: torch.dtype, device: str) -> PreTrainedModel:
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
    def special_token_ids(self) -> list[Any]:
        r"""Return the list of special tokens to exclude from the pooling."""
        ...

    @property
    @abstractmethod
    def mteb_model_meta(self) -> ModelMeta:
        r"""
        Metadata needed for MTEB v2. More information on the official documentation
        at https://github.com/embeddings-benchmark/mteb.
        """
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
    def _get_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        r"""
        Single forward pass recovering the sequence of tokens of shape (N, L, E) after the specified layer and module.

        Parameters
        ----------
        input_ids: torch.Tensor of shape (N, L)
            Tokenized input on the same device as the model.
        attention_mask: torch.Tensor of shape (N, L)
            Padding mask on the same device as the model.

        Returns
        -------
        hidden_states: torch.Tensor of shape (N, L, E)
            Token embeddings before pooling.
        """
        embeddings: dict[str, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        # Valid layer and module
        if self.layer > self.n_layers:
            raise ValueError(f"Layer index {self.layer} is out of range for a model with {self.n_layers} layers.")
        if self.module is not None:
            if self.module not in TRANSFORMER_MODULES:
                raise ValueError(f"Invalid module '{self.module}'. Options are {TRANSFORMER_MODULES}.")

        # Recover embedding from the specified layer and module
        key = f"block_{self.layer}_{self.module}"
        adapter = self.block_adapter(self.blocks[self.layer])
        module_obj = adapter.module_map()[self.module]

        if self.module in adapter.pre_hooks:
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
        batch_size: int = 256,
    ) -> torch.Tensor:
        r"""
        Encode a flat list of sentences into mean-pooled embeddings.

        Parameters
        ----------
        sentences: list[str]
            List of input sentences or documents to encode.
        batch_size: int, default=256
            Number of sentences to process per forward pass.

        Returns
        -------
        embeddings: torch.Tensor of shape (N, E)
            Sentence embeddings stored on CPU.
        """
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            text = sentences[i : i + batch_size]

            # Recover encoding, input_ids and padding tokens (attention_mask)
            encoding = self.tokenizer(text=text, padding=True, truncation=True, return_tensors="pt").to(self.device)
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

    def encode(
        self,
        inputs: DataLoader[BatchedInput] | list[str],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        r"""
        Encode inputs compatible with MTEB v2 EncoderProtocol.

        Unpacks the DataLoader[BatchedInput] into a flat list of sentences,
        then encodes them in batches using _encode_sentences.

        Parameters
        ----------
        inputs: DataLoader[BatchedInput] | list[str]
            Batched inputs from MTEB containing text under the "text" key.
        task_metadata: TaskMetadata
            Metadata of the current MTEB task.
        hf_split: str
            Current dataset split.
        hf_subset: str
            Current dataset subset.
        prompt_type: PromptType | None
            Query or passage prompt type.
        **kwargs: Unpack[EncodeKwargs]
            Additional arguments to pass to the encoder.

        Returns
        -------
        embeddings: torch.Tensor of shape (N, E)
            Sentence embeddings stored on CPU.
        """
        if isinstance(inputs, DataLoader):
            all_embeddings = []
            for batch in inputs:
                batch_embeddings = self._encode_sentences(sentences=batch["text"])
                all_embeddings.append(batch_embeddings)
            return torch.cat(all_embeddings, dim=0)

        elif isinstance(inputs, list):
            return self._encode_sentences(inputs)
        else:
            raise ValueError("Unsupported input type. Expected DataLoader or list of strings.")

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        r"""
        Cosine similarity between paired embeddings.

        Required by EncoderProtocol. Implemented here at the base class level so
        all subclasses (e.g. Gemma3) inherit it for free without boilerplate.
        Cosine similarity is consistent with similarity_fn_name="cosine" in ModelMeta.

        Parameters
        ----------
        embeddings1: Array of shape (N, E) or (E,)
            First set of embeddings.
        embeddings2: Array of shape (N, E) or (E,)
            Second set of embeddings.

        Returns
        -------
        Array of shape (N, M)
            Cosine similarity matrix.
        """
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        embeddings1 = F.normalize(embeddings1, dim=-1)
        embeddings2 = F.normalize(embeddings2, dim=-1)
        return embeddings1 @ embeddings2.T

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array:
        r"""
        Pairwise cosine similarity between corresponding embedding pairs.

        Required by EncoderProtocol. Implemented here at the base class level so
        all subclasses (e.g. Gemma3) inherit it for free without boilerplate.
        Cosine similarity is consistent with similarity_fn_name="cosine" in ModelMeta.

        Parameters
        ----------
        embeddings1: Array of shape (N, E) or (E,)
            First set of embeddings.
        embeddings2: Array of shape (N, E) or (E,)
            Second set of embeddings, paired with embeddings1.

        Returns
        -------
        Array of shape (N,)
            Cosine similarity for each pair.
        """
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        return F.cosine_similarity(embeddings1, embeddings2, dim=-1)

    def set_probing(self, layer: int | None, module: str | None) -> None:
        r"""
        Set the layer and module to probe for embedding extraction.

        Allows reusing the same model instance across multiple (layer, module) pairs without
        reloading weights. To keep the previous value for either argument, pass None.

        Parameters
        ----------
        layer: int
            Index of the layer index to probe.
        module: str
            Module name from TRANSFORMER_MODULES to probe.
        """
        # Valid layer and module
        if layer is not None:
            if layer > self.n_layers:
                raise ValueError(f"Layer index {layer} is out of range for a model with {self.n_layers} layers.")
            self.layer = layer
        if module is not None:
            if module not in TRANSFORMER_MODULES:
                raise ValueError(f"Invalid module '{module}'. Options are {TRANSFORMER_MODULES}.")
            self.module = module

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
