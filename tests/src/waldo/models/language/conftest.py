r"""
Shared fixtures for large language models unit tests.

Models are built from tiny randomly initialized HuggingFace models with the architecture of the pretrained ones,
and a whitespace tokenizer, so that tests run without network access or pretrained weights.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import (
    BatchEncoding,
    Gemma3TextConfig,
    Gemma3TextModel,
    LlamaConfig,
    LlamaModel,
    PretrainedConfig,
    PreTrainedModel,
    Qwen3Config,
    Qwen3Model,
)

from waldo.models import BaseLanguageModel, build_model

# Default parameters
N_LAYERS = 3
HIDDEN_SIZE = 32
VOCAB_SIZE = 100
TINY = dict(
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=N_LAYERS,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    intermediate_size=64,
    vocab_size=VOCAB_SIZE,
)
SENTENCES = ["The cat sits on the mat", "Hello", "A longer sentence with more words in it"]

# Type of the make_language_model fixture
MakeLanguageModel = Callable[..., tuple[BaseLanguageModel, PreTrainedModel]]


class WhitespaceTokenizer:
    r"""Tokenizer splitting on whitespaces, adding a BOS token and padding on the right."""

    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    pad_token, bos_token, eos_token = "<pad>", "<bos>", "<eos>"

    def __call__(
        self, text: list[str], padding: bool = True, truncation: bool = True, return_tensors: str = "pt"
    ) -> BatchEncoding:
        ids = [[self.bos_token_id] + [3 + sum(map(ord, word)) % (VOCAB_SIZE - 3) for word in s.split()] for s in text]
        length = max(len(x) for x in ids)
        input_ids = torch.tensor([x + [self.pad_token_id] * (length - len(x)) for x in ids])
        attention_mask = torch.tensor([[1] * len(x) + [0] * (length - len(x)) for x in ids])
        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})


@dataclass
class LanguageSpec:
    r"""Tiny HuggingFace model standing for a pretrained model, and the waldo config that loads it."""

    hf_class: type[PreTrainedModel]
    hf_config: PretrainedConfig
    config: dict[str, Any]


LANGUAGE_SPECS = {
    "gemma3": LanguageSpec(Gemma3TextModel, Gemma3TextConfig(**TINY), {"model_name": "gemma3", "model_size": "1b"}),
    "llama3": LanguageSpec(LlamaModel, LlamaConfig(**TINY), {"model_name": "llama3", "model_size": "1B"}),
    "qwen3": LanguageSpec(Qwen3Model, Qwen3Config(**TINY), {"model_name": "qwen3", "model_size": "0.6B"}),
}


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------
@pytest.fixture
def make_language_model() -> MakeLanguageModel:
    r"""Build a waldo language model on top of the tiny HuggingFace model of a spec, and return both."""

    def _factory(name: str, **config: Any) -> tuple[BaseLanguageModel, PreTrainedModel]:
        spec = LANGUAGE_SPECS[name]
        torch.manual_seed(0)
        hf_model = spec.hf_class(spec.hf_config).eval()
        with (
            patch.object(spec.hf_class, "from_pretrained", return_value=hf_model),
            patch(f"waldo.models.language.{name}.AutoTokenizer.from_pretrained", return_value=WhitespaceTokenizer()),
        ):
            model = build_model({**spec.config, "device": "cpu", **config})
        return model, hf_model

    return _factory


@pytest.fixture(params=list(LANGUAGE_SPECS))
def language_spec(request: pytest.FixtureRequest) -> str:
    r"""Name of each language spec, to run a test on all architectures."""
    return request.param
