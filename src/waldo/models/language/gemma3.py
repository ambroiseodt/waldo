r"""
Gemma3 pretrained model. It allows to study specific layers and modules following [1, 2]_.

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

Notes
-----
We focus on the pretrained models available on HuggingFace for text generation.
See https://huggingface.co/collections/google/gemma-3 for other other variants.

References
----------
.. [1] Gemma Team. Gemma 3 Technical Report. arXiv 2025

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

import torch
import torch.nn as nn
from torch.nn import ModuleList
from transformers import AutoTokenizer, Gemma3TextModel, PreTrainedModel, PreTrainedTokenizerBase

from ..transformer_block import TransformerBlockAdapter
from ..utils import mean_pooling
from .base import BaseLanguageModel

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
#   Config
# -------------------------------------------------------------------
@dataclass
class Gemma3Config:
    r"""
    Gemma3 configuration file.

    Parameters
    ----------
    model_size: str
        Size of the model. Options are "1b", "4b", "12b", "27b".
    dtype: torch.dtype
        Dtype to load the model weights in. Defaults to bfloat16 since Gemma3 was
        pretrained in bfloat16, which halves memory footprint vs float32 with no
        accuracy loss. Use float16 for older GPUs that do not support bfloat16,
        or float32 for CPU inference or when full precision is required.
    device: str
        Device to use.
    """

    model_size: str = "1b"
    dtype: torch.dtype = torch.bfloat16
    device: str = "cpu"

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.model_size not in ["1b", "4b", "12b", "27b"]:
            raise ValueError(f"Invalid model size '{self.model_size}'. Options are '1b', '4b', '12b', '27b'.")


# -------------------------------------------------------------------
#   Transformer block adapter
# -------------------------------------------------------------------
class Gemma3BlockAdapter(TransformerBlockAdapter):
    r"""
    Adapter mapping standard module names to Gemma3DecoderLayer submodules.

    Gemma3 uses a pre-norm architecture with SwiGLU in the FFN. For consistency
    with other models, we decompose the FFN block as follows:

        fc1  = gate_proj(x)                      -> linear layer which feeds the activation
        act  = silu(gate_proj(x)) * up_proj(x)   -> non-linearity (which is a function of the input too)
        fc2  = down_proj(act)                    -> linear layer which gives the output of the FFN

    The full module mapping is:

        ln1  -> input_layernorm
        mha  -> self_attn
        rc1  -> pre_feedforward_layernorm  (pre-hook: residual after attention = input to FFN branch)
        ln2  -> pre_feedforward_layernorm
        fc1  -> mlp.gate_proj
        act  -> mlp.down_proj              (pre-hook: gated activation output = input to down_proj)
        fc2  -> mlp.down_proj
        rc2  -> post_feedforward_layernorm (pre-hook: residual after FFN = input to post norm)

    Notes
    -----
    rc1, act, and rc2 are declared as pre-hook modules following the convention in
    TransformerBlockAdapter: their outputs are recovered as inputs to the next submodule
    rather than outputs of the current one.

    See also
    --------
    TransformerBlockAdapter
    """

    def _get_ln1(self, layer: nn.Module) -> nn.Module:
        return layer.input_layernorm

    def _get_mha(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn

    def _get_rc1(self, layer: nn.Module) -> nn.Module:
        return layer.pre_feedforward_layernorm

    def _get_ln2(self, layer: nn.Module) -> nn.Module:
        return layer.pre_feedforward_layernorm

    def _get_fc1(self, layer: nn.Module) -> nn.Module:
        return layer.mlp.gate_proj

    def _get_act(self, layer: nn.Module) -> nn.Module:
        return layer.mlp.down_proj

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        return layer.mlp.down_proj

    def _get_rc2(self, layer: nn.Module) -> nn.Module:
        return layer.post_feedforward_layernorm

    def _get_pre_hooks_modules(self, layer: nn.Module) -> set[str]:
        r"""
        Returns the names of the module for which register_forward_pre_hook should be used
        in place of register_forward_hook to access their inputs instead of the outputs.
        """
        return {"rc1", "act", "rc2"}


# -------------------------------------------------------------------
#   Model
# -------------------------------------------------------------------
class Gemma3(BaseLanguageModel):
    r"""
    Gemma3 language model following [1]_.

    Parameters
    ----------
    config: Gemma3Config
        Configuration object specifying model size, layer, module, and device.

    Notes
    -----
    The model is loaded in evaluation mode by default (dropout disabled).
    Call model.train() explicitly if you intend to fine-tune.

    References
    ----------
    .. [1] Gemma Team. Gemma 3 Technical Report. 2025
    """

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.device = config.device
        self.model_name = f"google/gemma-3-{config.model_size}-pt"
        self.dtype = config.dtype
        self.model = self._load_from_huggingface(model_name=self.model_name, dtype=self.dtype, device=self.device)
        self.tokenizer = self._load_tokenizer(model_name=self.model_name)

    def _load_from_huggingface(self, model_name: str, dtype: torch.dtype, device: str) -> PreTrainedModel:
        r"""
        Load weights from the HuggingFace Transformers library.

        Parameters
        ----------
        model_name: str
            Name of the model. Options are:
                "google/gemma-3-1b-pt",
                "google/gemma-3-4b-pt",
                "google/gemma-3-12b-pt",
                "google/gemma-3-27b-pt".
        dtype: torch.dtype
            Dtype to load the model weights in. Defaults to bfloat16 since Gemma3 was
            pretrained in bfloat16, which halves memory footprint vs float32 with no
            accuracy loss. Use float16 for older GPUs that do not support bfloat16,
            or float32 for CPU inference or when full precision is required.
        device: str
            Device to use.

        Notes
        -----
        The model is set to evaluation mode after loading (dropout disabled).
        Call model.train() explicitly to re-enable training mode.
        The model is loaded in bfloat16 to reduce memory footprint.
        """
        available_models = [
            "google/gemma-3-1b-pt",
            "google/gemma-3-4b-pt",
            "google/gemma-3-12b-pt",
            "google/gemma-3-27b-pt",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Available models: {available_models}")
        model = Gemma3TextModel.from_pretrained(
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
        return Gemma3BlockAdapter

    @property
    def blocks(self) -> ModuleList:
        r"""Return the list of transformer blocks."""
        return self.model.model.layers

    @property
    def special_token_ids(self) -> list[int | None]:
        r"""Return BOS, EOS and padding token ids to exclude from pooling."""
        return [
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        ]
