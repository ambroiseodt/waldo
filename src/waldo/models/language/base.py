r"""
Abstract model for large language models (LLMs). It allows to study specific layers and modules
and exposes an encode() method compatible with the MTEB benchmark.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension

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

from .base import TransformerBlockAdapter


# -------------------------------------------------------------------
# Base Language Model
# -------------------------------------------------------------------
class BaseLanguageModel(nn.Module, ABC):
    r"""
    Abstract base class for transformer-based language foundation models.

    It handles tokenization, sentence mask construction, hook registration,
    and mean pooling in a single forward pass. Subclasses only need to implement:
        - _load_from_huggingface: load and return the HuggingFace model
        - _load_tokenizer: load and return the tokenizer
        - block_adapter: return the adapter class for this model's blocks
        - blocks: return the ModuleList of transformer blocks

    The encode() method is MTEB-compatible — it accepts a list of sentences and
    returns mean-pooled embeddings from the specified layer and module, excluding
    padding, BOS, and EOS tokens from the pooling.

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled). Call model.train()
    explicitly if you intend to fine-tune.

    Unlike BaseVisionModel, hooks capture the full (N, L, E) tensor rather than
    pooling inside the hook. This is necessary to apply the sentence mask after the
    forward pass, since the mask is only available at encode() time.
    """

    def __init__(self):
        super().__init__()
        self.device: str
        self.model: PreTrainedModel
        self.tokenizer: PreTrainedTokenizerBase
        self.layer: int
        self.module: str

    @abstractmethod
    def _load_from_huggingface(self, model_name: str, device: str) -> PreTrainedModel:
        r"""Load a pretrained HuggingFace model in evaluation mode on the specified device."""
        ...

    @abstractmethod
    def _load_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        r"""Load the tokenizer for the model."""
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
        Create a forward pre-hook that captures the full INPUT tensor of a module.

        Unlike BaseVisionModel, no pooling is applied inside the hook — the full
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

        Unlike BaseVisionModel, no pooling is applied inside the hook — the full
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

    def _build_sentence_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Build a mask that is 1 only for true sentence tokens.

        Excludes padding (attention_mask == 0), BOS, and EOS tokens.

        Parameters
        ----------
        input_ids: torch.Tensor of shape (N, L)
            Token ids of the input sequences.
        attention_mask: torch.Tensor of shape (N, L)
            Padding mask from the tokenizer (1 = real token, 0 = padding).

        Returns
        -------
        sentence_mask: torch.Tensor of shape (N, L)
            Binary mask with 1 only for true sentence tokens.
        """
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        sentence_mask = attention_mask.clone()
        if bos_id is not None:
            sentence_mask = sentence_mask * (input_ids != bos_id).long()
        if eos_id is not None:
            sentence_mask = sentence_mask * (input_ids != eos_id).long()

        return sentence_mask

    def _mean_pooling(
        self,
        hidden_states: torch.Tensor,
        sentence_mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Mean pool over true sentence tokens.

        Parameters
        ----------
        hidden_states: torch.Tensor of shape (N, L, E)
            Token embeddings captured by a hook.
        sentence_mask: torch.Tensor of shape (N, L)
            Binary mask with 1 only for true sentence tokens.

        Returns
        -------
        pooled: torch.Tensor of shape (N, E)
            Mean-pooled sentence embeddings stored on CPU.
        """
        mask = sentence_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return (sum_embeddings / sum_mask).cpu()

    @torch.inference_mode()
    def _get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: int,
        module: str,
    ) -> torch.Tensor:
        r"""
        Single forward pass recovering the (N, L, E) tensor from one layer and module.

        Parameters
        ----------
        input_ids: torch.Tensor of shape (N, L)
            Tokenized input on the model device.
        attention_mask: torch.Tensor of shape (N, L)
            Padding mask on the model device.
        layer: int
            Layer index to recover.
        module: str
            Module name from TRANSFORMER_MODULES to recover.

        Returns
        -------
        hidden_states: torch.Tensor of shape (N, L, E)
            Raw token embeddings before pooling.
        """
        embeddings: dict[str, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []
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

        return embeddings[key]

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Encode a list of sentences into embeddings compatible with MTEB.

        Tokenizes the input, runs a forward pass with a hook on the specified
        layer and module, and returns mean-pooled embeddings over true sentence
        tokens only (excluding padding, BOS, EOS).

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
        layer = self.layer % self.n_layers

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            sentence_mask = self._build_sentence_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            hidden_states = self._get_embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer=layer,
                module=self.module,
            )

            pooled = self._mean_pooling(
                hidden_states=hidden_states,
                sentence_mask=sentence_mask,
            )
            all_embeddings.append(pooled)

        return torch.cat(all_embeddings, dim=0)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return (
            f"{self.__class__.__name__} — layer={self.layer}, module={self.module}, {n_params:,} trainable parameters."
        )

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers
