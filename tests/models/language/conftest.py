r"""
Shared fixtures for large language models unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from waldo.models import Gemma3, Gemma3Config, Gemma4, Gemma4Config, Llama3, Llama3Config, Qwen3, Qwen3Config

# Default parameters
N_LAYERS = 3
HIDDEN_SIZE = 16
BATCH_SIZE = 2
SEQ_LEN = 10


# ---------------------------------------------------------------------------
#  Moke transformer modules
# ---------------------------------------------------------------------------
class MockLinear(nn.Linear):
    r"""A mock linear layer whose forward is the identity."""

    def __init__(self, dim: int = HIDDEN_SIZE):
        super().__init__(dim, dim, bias=False)
        nn.init.eye_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


class MockRMSNorm(nn.RMSNorm):
    r"""A mock RMSNorm."""

    def __init__(self, dim: int = HIDDEN_SIZE):
        super().__init__(dim)


class MockActivation(nn.Module):
    r"""A mock activation returning ReLU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class MockAttention(nn.Module):
    r"""A mock attention layer whose forward is the identity."""

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, None]:
        return x, None


# ---------------------------------------------------------------------------
# Moke transformer blocks
# ---------------------------------------------------------------------------
class MockGatedMLP(nn.Module):
    r"""Mocks gated activation MLP like SwiGLU."""

    def __init__(self):
        nn.Module.__init__(self)
        self.gate_proj = MockLinear()
        self.up_proj = MockLinear()
        self.act_fn = MockActivation()
        self.down_proj = MockLinear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MockGemma3Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = MockRMSNorm()
        self.self_attn = MockAttention()
        self.pre_feedforward_layernorm = MockRMSNorm()
        self.mlp = MockGatedMLP()

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        attn_out, _ = self.self_attn(self.input_layernorm(x))
        rc1 = x + attn_out
        ffn_out = rc1 + self.mlp(self.pre_feedforward_layernorm(rc1))
        return (ffn_out,)


class MockGemma4Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = MockRMSNorm()
        self.self_attn = MockAttention()
        self.pre_feedforward_layernorm = MockRMSNorm()
        self.mlp = MockGatedMLP()

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        attn_out, _ = self.self_attn(self.input_layernorm(x))
        rc1 = x + attn_out
        ffn_out = rc1 + self.mlp(self.pre_feedforward_layernorm(rc1))
        return (ffn_out,)


class MockLlama3Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = MockRMSNorm()
        self.self_attn = MockAttention()
        self.post_attention_layernorm = MockRMSNorm()
        self.mlp = MockGatedMLP()

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        attn_out, _ = self.self_attn(self.input_layernorm(x))
        rc1 = x + attn_out
        ffn_out = rc1 + self.mlp(self.post_attention_layernorm(rc1))
        return (ffn_out,)


class MockQwen3Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = MockRMSNorm()
        self.self_attn = MockAttention()
        self.post_attention_layernorm = MockRMSNorm()
        self.mlp = MockGatedMLP()

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        attn_out, _ = self.self_attn(self.input_layernorm(x))
        rc1 = x + attn_out
        ffn_out = rc1 + self.mlp(self.post_attention_layernorm(rc1))
        return (ffn_out,)


# ---------------------------------------------------------------------------
# Global fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def input_ids() -> torch.Tensor:
    r"""A random batch of input_ids representative of tokenized text."""
    return torch.randint(low=10, high=100, size=(BATCH_SIZE, SEQ_LEN))


@pytest.fixture(scope="session")
def attention_mask() -> torch.Tensor:
    r"""A mock attention mask (all ones, no padding)."""
    return torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)


@pytest.fixture(scope="session")
def make_mock_language_model() -> Callable[..., tuple[MagicMock, nn.ModuleList]]:
    r"""A mock model factory for language models inheriting from BaseLanguageModel."""

    def _make_language_model_shell(
        blocks: list[nn.Module], hidden_size: int = HIDDEN_SIZE
    ) -> tuple[MagicMock, nn.ModuleList]:

        layers = nn.ModuleList(blocks)

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = hidden_size
        mock_model.eval.return_value = mock_model
        mock_model.parameters.return_value = iter([])
        mock_model.config.max_position_embeddings = 512

        def _forward(input_ids: torch.Tensor, **kwargs) -> MagicMock:
            x = torch.zeros(input_ids.shape[0], input_ids.shape[1], hidden_size)
            for block in layers:
                (x,) = block(x)
            out = MagicMock()
            out.last_hidden_state = x

            # Ensure indexing works for hooks extracting output[0]
            out.__getitem__.side_effect = lambda idx: x if idx == 0 else None

            return out

        # Allow to run the mock blocks to fire the hook and pre-hook calls
        mock_model.side_effect = _forward

        # Ensure robust call
        mock_model.__call__ = _forward

        return mock_model, layers

    return _make_language_model_shell


# -----------------------------------------------------------------------------
# Dataset-specific fixtures
# ------------------------------------------------------------------------------
@pytest.fixture
def mock_gemma3_model(make_mock_language_model: Callable[..., tuple[MagicMock, nn.ModuleList]]) -> Gemma3:
    r"""
    A Gemma3 instance whose HuggingFace backbone is fully mocked.
    Patches Gemma3TextModel.from_pretrained and AutoTokenizer so no network call is made.
    """

    blocks = [MockGemma3Block() for _ in range(N_LAYERS)]
    mock_hf, layers = make_mock_language_model(blocks)
    mock_hf.layers = layers

    with (
        patch("waldo.models.language.gemma3.Gemma3TextModel.from_pretrained", return_value=mock_hf),
        patch("waldo.models.language.gemma3.AutoTokenizer.from_pretrained") as mock_tok,
    ):
        # Setup mock tokenizer with standard special tokens
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.bos_token_id = 1
        mock_tokenizer.eos_token_id = 2
        mock_tok.return_value = mock_tokenizer

        config = Gemma3Config(model_size="1b", device="cpu")
        model = Gemma3(config)

    model.model = mock_hf
    return model


@pytest.fixture
def mock_gemma4_model(make_mock_language_model: Callable[..., tuple[MagicMock, nn.ModuleList]]) -> Gemma4:
    r"""
    A Gemma4 instance whose HuggingFace backbone is fully mocked.
    Patches Gemma4TextModel.from_pretrained and AutoTokenizer so no network call is made.
    """

    blocks = [MockGemma4Block() for _ in range(N_LAYERS)]
    mock_hf, layers = make_mock_language_model(blocks)
    mock_hf.layers = layers

    with (
        patch("waldo.models.language.gemma4.Gemma4TextModel.from_pretrained", return_value=mock_hf),
        patch("waldo.models.language.gemma4.AutoTokenizer.from_pretrained") as mock_tok,
    ):
        # Setup mock tokenizer with standard special tokens
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.bos_token_id = 1
        mock_tokenizer.eos_token_id = 2
        mock_tok.return_value = mock_tokenizer

        config = Gemma4Config(model_size="E2B", device="cpu")
        model = Gemma4(config)

    model.model = mock_hf
    return model


@pytest.fixture
def mock_llama3_model(make_mock_language_model: Callable[..., tuple[MagicMock, nn.ModuleList]]) -> Llama3:
    r"""
    A Llama3 instance whose HuggingFace backbone is fully mocked.
    Patches LlamaModel.from_pretrained and AutoTokenizer so no network call is made.
    """

    blocks = [MockLlama3Block() for _ in range(N_LAYERS)]
    mock_hf, layers = make_mock_language_model(blocks)
    mock_hf.layers = layers

    with (
        patch("waldo.models.language.llama3.LlamaModel.from_pretrained", return_value=mock_hf),
        patch("waldo.models.language.llama3.AutoTokenizer.from_pretrained") as mock_tok,
    ):
        # Setup mock tokenizer with standard special tokens
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.bos_token_id = 1
        mock_tokenizer.eos_token_id = 2
        mock_tok.return_value = mock_tokenizer

        config = Llama3Config(model_size="3.2-1B", device="cpu")
        model = Llama3(config)

    model.model = mock_hf
    return model


@pytest.fixture
def mock_qwen3_model(make_mock_language_model: Callable[..., tuple[MagicMock, nn.ModuleList]]) -> Qwen3:
    r"""
    A Qwen3 instance whose HuggingFace backbone is fully mocked.
    Patches Qwen3Model.from_pretrained and AutoTokenizer so no network call is made.
    """

    blocks = [MockQwen3Block() for _ in range(N_LAYERS)]
    mock_hf, layers = make_mock_language_model(blocks)
    mock_hf.layers = layers

    with (
        patch("waldo.models.language.qwen3.Qwen3Model.from_pretrained", return_value=mock_hf),
        patch("waldo.models.language.qwen3.AutoTokenizer.from_pretrained") as mock_tok,
    ):
        # Setup mock tokenizer with standard special tokens
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.bos_token_id = 1
        mock_tokenizer.eos_token_id = 2
        mock_tok.return_value = mock_tokenizer

        config = Qwen3Config(model_size="0.6B", device="cpu")
        model = Qwen3(config)

    model.model = mock_hf
    return model
