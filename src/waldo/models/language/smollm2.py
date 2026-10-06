r"""
SmolLM2 pretrained model following [1]_. It allows to study specific layers and modules.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension

Notes
-----
We focus on the pretrained (base) models available on HuggingFace for text generation.
See https://huggingface.co/collections/HuggingFaceTB/smollm2 for variants.

References
----------
.. [1] L. Ben Allal et al. SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model.
       arXiv 2025

See also
--------
TransformerBlockAdapter
BaseLanguageModel

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
from mteb.models.model_meta import ModelMeta, ScoringFunction
from torch.nn import ModuleList
from transformers import AutoTokenizer, LlamaModel, PreTrainedModel, PreTrainedTokenizerBase
from transformers.models.llama.modeling_llama import LlamaMLP

from ..pooling import mean_pooling
from ..transformer_block import TransformerBlockAdapter
from .base import BaseLanguageModel

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
#   Config
# -------------------------------------------------------------------
@dataclass
class SmolLM2Config:
    r"""
    SmolLM2 configuration file.

    Parameters
    ----------
    model_size: str
        Size of the model. Options are "135M", "360M", "1.7B".
    dtype: torch.dtype
        Dtype to load the model weights in. Defaults to bfloat16 since SmolLM2 was
        pretrained in bfloat16, which halves the memory footprint compared with float32.
        Use float16 for older GPUs that do not support bfloat16, or float32 when full precision
        is required.
    device: str
        Device to use.
    """

    model_size: str = "135M"
    dtype: torch.dtype = torch.bfloat16
    device: str = "cpu"

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.model_size not in ["135M", "360M", "1.7B"]:
            raise ValueError(f"Invalid model size '{self.model_size}'. Options are '135M', '360M', '1.7B'.")


# -------------------------------------------------------------------
#   Transformer block adapter
# -------------------------------------------------------------------
class SmolLM2BlockAdapter(TransformerBlockAdapter):
    r"""
    Adapter mapping the standard module names to the submodules of a SmolLM2 transformer block.

    Parameters
    ----------
    layer: nn.Module
        A single SmolLM2 transformer block.

    Notes
    -----
    SmolLM2 relies on the Llama architecture, with RMSNorm and a SwiGLU activation in the FFN [1, 2, 3]_: act is the
    input of the down projection, following the decomposition of TransformerBlockAdapter.

    References
    ----------
    .. [1] P. Ramachandran et al. Searching for activation functions. arXiv 2017

    .. [2] Y. Dauphin et al. Language Modeling with Gated Convolutional Networks. In ICML 2017

    .. [3] N. Shazeer. GLU Variants Improve Transformer. arXiv 2020

    See also
    --------
    TransformerBlockAdapter
    """

    def _get_ln1(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.input_layernorm)

    def _get_mha(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.self_attn)

    def _get_rc1(self, layer: nn.Module) -> nn.Module:
        # Use pre_hook to capture rc1: residual after attention, before FFN branch
        return cast(nn.Module, layer.post_attention_layernorm)

    def _get_ln2(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, layer.post_attention_layernorm)

    def _get_fc1(self, layer: nn.Module) -> nn.Module:
        ffn = cast(LlamaMLP, layer.mlp)
        return cast(nn.Module, ffn.gate_proj)

    def _get_act(self, layer: nn.Module) -> nn.Module:
        # Use pre_hook to capture act: gated activation output = input to down_proj
        ffn = cast(LlamaMLP, layer.mlp)
        return cast(nn.Module, ffn.down_proj)

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        ffn = cast(LlamaMLP, layer.mlp)
        return cast(nn.Module, ffn.down_proj)

    def _get_rc2(self, layer: nn.Module) -> nn.Module:
        return layer

    def _get_pre_hooks_modules(self, layer: nn.Module) -> set[str]:
        r"""
        Returns the names of the modules for which register_forward_pre_hook should be used
        in place of register_forward_hook to access their inputs instead of their outputs.
        """
        return {"rc1", "act"}


# -------------------------------------------------------------------
#   Model
# -------------------------------------------------------------------
class SmolLM2(BaseLanguageModel):
    r"""
    SmolLM2 language model following [1]_.

    By default, the probing is done on the output of the final transformer block.
    Set the layer and module to probe with set_probing(layer: int, module: str).
    See base.py for more details.

    Parameters
    ----------
    config: configuration class with
        model_size: str
            Size of the model. Options are "135M", "360M", "1.7B".
        dtype: torch.dtype
            Dtype to load the model weights in. Defaults to bfloat16 since SmolLM2 was
            pretrained in bfloat16, which halves the memory footprint compared with float32.
            Use float16 for older GPUs that do not support bfloat16, or float32 when full precision
            is required.
        device: str
            Device to use.

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled). Call `model.train()`
    explicitly if you intend to fine-tune.
    The model is loaded with LlamaModel since SmolLM2 relies on the Llama architecture.

    References
    ----------
    .. [1] L. Ben Allal et al. SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model.
           arXiv 2025
    """

    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.device = config.device
        self.model_name = f"HuggingFaceTB/SmolLM2-{config.model_size}"
        self.dtype = config.dtype
        self.model = self._load_model(model_name=self.model_name, dtype=self.dtype, device=self.device)
        self.tokenizer = self._load_tokenizer(model_name=self.model_name)

        # SmolLM2 tokenizer does not have a pad token. Following standard practices, we set it equal to eos token.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def _load_model(self, model_name: str, dtype: torch.dtype, device: str) -> PreTrainedModel:
        r"""
        Load weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "HuggingFaceTB/SmolLM2-135M",
                "HuggingFaceTB/SmolLM2-360M",
                "HuggingFaceTB/SmolLM2-1.7B".
        dtype: torch.dtype
            Dtype to load the model weights in. Defaults to bfloat16 since SmolLM2 was
            pretrained in bfloat16, which halves the memory footprint compared with float32.
            Use float16 for older GPUs that do not support bfloat16, or float32 when full precision
            is required.
        device: str
            Device to use.

        Notes
        -----
        The model is loaded in evaluation mode by default (dropout disabled). Call `model.train()`
        explicitly if you intend to fine-tune.
        The model is loaded with LlamaModel since SmolLM2 relies on the Llama architecture.
        """
        available_models = [
            "HuggingFaceTB/SmolLM2-135M",
            "HuggingFaceTB/SmolLM2-360M",
            "HuggingFaceTB/SmolLM2-1.7B",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")
        model = LlamaModel.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
        )
        model.eval()
        return model

    def _load_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        r"""
        Load the tokenizer from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model.
        """
        return AutoTokenizer.from_pretrained(model_name)

    def pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        special_token_ids: list[int | None] | None = None,
    ) -> torch.Tensor:
        r"""
        Mean pooling over true sentence tokens stored on CPU.

        Excludes padding, BOS, and EOS tokens from pooling. Since embeddings are
        obtained from _get_hidden_states which uses @torch.inference_mode(),
        there is no need to detach the tensor.

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
            Mean-pooled sentence embedding stored on CPU.
        """
        return mean_pooling(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            input_ids=input_ids,
            special_token_ids=special_token_ids,
        ).cpu()

    @property
    def block_adapter(self) -> type[TransformerBlockAdapter]:
        r"""Return the block adapter class for the model's transformer blocks."""
        return SmolLM2BlockAdapter

    @property
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        return cast(ModuleList, self.model.layers)

    @property
    def final_norm(self) -> nn.Module:
        r"""Return the final norm applied to the residual stream of the last transformer block."""
        return cast(nn.Module, self.model.norm)

    @property
    def special_token_ids(self) -> list[Any]:
        r"""
        Return BOS, EOS and padding token ids to exclude from pooling.

        Notes
        -----
        Since we consider pretrained models and not instruction-tuned variants,
        we do not expect special tokens related to multi-turn or thinking. As such,
        we only exclude separation and padding tokens. In SmolLM2, the BOS, EOS and
        padding tokens are the same <|endoftext|> token.
        """
        return [
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        ]

    @property
    def mteb_model_meta(self) -> ModelMeta:
        r"""
        Metadata for SmolLM2. It encompasses the layer and module used to encode text.

        Fields with None values are either not applicable (e.g. training datasets for a proprietary model)
        or not publicly available (e.g. memory usage, training code, training data).
        More information at https://arxiv.org/pdf/2502.02737.
        """

        # Ensure positive layer index
        layer = self.layer + self.n_layers if self.layer < 0 else self.layer

        return ModelMeta(
            name=f"{self.model_name}_layer_{layer}_module_{self.module}",
            revision="main",
            release_date="2024-10-31",
            open_weights=True,
            framework=["PyTorch"],
            license="apache-2.0",
            languages=None,
            loader=None,
            memory_usage_mb=None,
            training_datasets=None,
            public_training_code=None,
            public_training_data=None,
            use_instructions=False,
            similarity_fn_name=ScoringFunction.COSINE,
            n_parameters=sum(p.numel() for p in self.model.parameters()),
            max_tokens=self.model.config.max_position_embeddings,
            embed_dim=self.embedding_dim,
            modalities=["text"],
            adapted_from=None,
            superseded_by=None,
        )
