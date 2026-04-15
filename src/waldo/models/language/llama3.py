r"""
Llama3 pretrained model following [1]_. It allows to study specific layers and modules.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension

Notes
-----
We focus on the pretrained dense models available on HuggingFace for text generation.
See https://huggingface.co/collections/meta-llama/llama-31 (resp. llama-32) for variants.

References
----------
.. [1] A. Grattafiori et al. The Llama 3 Herd of Models. arXiv 2024

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

from ..transformer_block import TransformerBlockAdapter
from ..utils import mean_pooling
from .base import BaseLanguageModel

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
#   Config
# -------------------------------------------------------------------
@dataclass
class Llama3Config:
    r"""
    Llama3 configuration file.

    Parameters
    ----------
    model_size: str
        Size of the model. Options are "3.2-1B", "3.2-3B", "3.1-8B", "3.1-70B".
    dtype: torch.dtype
        Dtype to load the model weights in. Defaults to bfloat16 since Llama3 was
        pretrained in bfloat16, which halves memory footprint vs float32 with no
        accuracy loss. Use float16 for older GPUs that do not support bfloat16,
        or float32 for CPU inference or when full precision is required.
    device: str
        Device to use.
    """

    model_size: str = "3.2-1B"
    dtype: torch.dtype = torch.bfloat16
    device: str = "cpu"

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.model_size not in ["3.2-1B", "3.2-3B", "3.1-8B", "3.1-70B"]:
            raise ValueError(
                f"Invalid model size '{self.model_size}'. Options are '3.2-1B], '3.2-3B', '3.1-8B', '3.1-70B'."
            )


# -------------------------------------------------------------------
#   Transformer block adapter
# -------------------------------------------------------------------
class Llama3BlockAdapter(TransformerBlockAdapter):
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
    llama3 relies on a gated SwiGLU activation in the FFN (see [1, 2]_ for details). For consistency with other models,
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
    .. [1] Y. Dauphin et al. Language Modeling with Gated Convolutional Networks. In ICML 2017

    .. [2] N. Shazeer. GLU Variants Improve Transformer. arXiv 2020

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
        Returns the names of the module for which register_forward_pre_hook should be used
        in place of register_forward_hook to access their inputs instead of the outputs.
        """
        return {"rc1", "act"}


# -------------------------------------------------------------------
#   Model
# -------------------------------------------------------------------
class Llama3(BaseLanguageModel):
    r"""
    Llama3 language model following [1]_.

    Parameters
    ----------
    config: configuration class with
        model_size: str
            Size of the model. Options are "3.2-1B", "3.2-3B", "3.1-8B", "3.1-70B".
        dtype: torch.dtype
            Dtype to load the model weights in. Defaults to bfloat16 since Gemma3 was
            pretrained in bfloat16, which halves memory footprint vs float32 with no
            accuracy loss. Use float16 for older GPUs that do not support bfloat16,
            or float32 for CPU inference or when full precision is required.
        device: str
            Device to use.

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled, BatchNorm uses
    running statistics). Call `model.train()` explicitly if you intend to fine-tune.
    The model is loaded in bfloat16 to reduce memory footprint. Note that he model loading
    is done with LlamaModel because Llama, Llama2 and Llama3 share the same architecture.

    References
    ----------
    .. [1] A. Grattafiori et al. The Llama 3 Herd of Models. arXiv 2024
    """

    def __init__(self, config: Llama3Config):
        super().__init__()
        self.device = config.device
        self.model_name = f"meta-llama/Llama-{config.model_size.upper()}"
        self.dtype = config.dtype
        self.model = self._load_model(model_name=self.model_name, dtype=self.dtype, device=self.device)
        self.tokenizer = self._load_tokenizer(model_name=self.model_name)

        # Llama3 tokenizer does not have a pad token. Following standard practices, we set it equal to eos token.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def _load_model(self, model_name: str, dtype: torch.dtype, device: str) -> PreTrainedModel:
        r"""
        Load weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "meta-llama/Llama-3.2-1B",
                "meta-llama/Llama-3.2-3B",
                "meta-llama/Llama-3.1-8B",
                "meta-llama/Llama-3.1-70B".
        dtype: torch.dtype
            Dtype to load the model weights in. Defaults to bfloat16 since llama3 was
            pretrained in bfloat16, which halves memory footprint vs float32 with no
            accuracy loss. Use float16 for older GPUs that do not support bfloat16,
            or float32 for CPU inference or when full precision is required.
        device: str
            Device to use.

        Notes
        -----
        The model is loaded in evaluation mode by default (dropout disabled, BatchNorm uses
        running statistics). Call `model.train()` explicitly if you intend to fine-tune.
        The model is loaded in bfloat16 to reduce memory footprint. Note that he model loading
        is done with LlamaModel because Llama, Llama2 and Llama3 share the same architecture.
        """
        available_models = [
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-70B",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")
        model = LlamaModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
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
        return Llama3BlockAdapter

    @property
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        return cast(ModuleList, self.model.layers)

    @property
    def special_token_ids(self) -> list[Any]:
        r"""Return BOS, EOS and padding token ids to exclude from pooling."""
        return [
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        ]

    @property
    def mteb_model_meta(self) -> ModelMeta:
        r"""
        Metadata for Llama3. It encompasses the layer and module used to encode text.

        Fields with None values are either not applicable (e.g. training datasets for a proprietary model)
        or not publicly available (e.g. memory usage, training code, training data).
        More information at https://arxiv.org/pdf/2407.21783.
        """

        # Ensure positive layer index
        layer = self.layer + self.n_layers if self.layer < 0 else self.layer

        return ModelMeta(
            name=f"{self.model_name}_layer_{layer}_module_{self.module}",
            revision="main",
            release_date="2024-07-123" if "3.1" in self.model_name else "2024-09-25",
            open_weights=True,
            framework=["PyTorch"],
            license="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE"
            if "3.1" in self.model_name
            else "https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE",
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
