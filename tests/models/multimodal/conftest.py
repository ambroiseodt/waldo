r"""
Shared fixtures for vision language models unit tests.

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

from waldo.models import Clip, ClipConfig, Siglip2, Siglip2Config

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


class MockLayerNorm(nn.LayerNorm):
    r"""A mock LayerNorm."""

    def __init__(self, dim: int = HIDDEN_SIZE):
        super().__init__(dim)


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


class MockMLP(nn.Module):
    """A mock MLP using standard fc1, activation, fc2."""

    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = MockLinear()
        self.activation_fn = MockActivation()
        self.fc2 = MockLinear()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_state)))


# ---------------------------------------------------------------------------
# Moke transformer blocks
# ---------------------------------------------------------------------------
class MockClipBlock(nn.Module):
    r"""A mock CLIP transformer block."""

    def __init__(self):
        super().__init__()
        self.layer_norm1 = MockLayerNorm()
        self.self_attn = MockAttention()
        self.layer_norm2 = MockLayerNorm()
        self.mlp = MockMLP()

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        attn_out, _ = self.self_attn(self.layer_norm1(x))
        rc1 = x + attn_out
        ffn_out = rc1 + self.mlp(self.layer_norm2(rc1))
        return (ffn_out,)


class MockSiglip2Block(nn.Module):
    r"""A mock SigLip2 transformer block."""

    def __init__(self):
        super().__init__()
        self.layer_norm1 = MockLayerNorm()
        self.self_attn = MockAttention()
        self.layer_norm2 = MockLayerNorm()
        self.mlp = MockMLP()

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        attn_out, _ = self.self_attn(self.layer_norm1(x))
        rc1 = x + attn_out
        ffn_out = rc1 + self.mlp(self.layer_norm2(rc1))
        return (ffn_out,)


# ---------------------------------------------------------------------------
# Global fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pixel_values() -> torch.Tensor:
    r"""A random batch of pixel values representative of a real dataset sample."""
    return torch.randn(BATCH_SIZE, 3, 224, 224)


@pytest.fixture(scope="session")
def make_mock_vision_language_model() -> Callable[..., tuple[MagicMock, MagicMock, nn.ModuleList]]:
    r"""A mock model for vision language models inheriting from BaseVisionLanguageModel."""

    def _make_vision_language_model_shell(
        blocks: list[nn.Module], hidden_size: int = HIDDEN_SIZE
    ) -> tuple[MagicMock, MagicMock, nn.ModuleList]:

        layers = nn.ModuleList(blocks)

        # Vision model
        mock_vision_model = MagicMock()
        mock_vision_model.config = MagicMock()
        mock_vision_model.config.hidden_size = hidden_size
        mock_vision_model.eval.return_value = mock_vision_model
        mock_vision_model.parameters.return_value = iter([])

        def _forward(pixel_values: torch.Tensor, **kwargs) -> MagicMock:
            x = torch.zeros(pixel_values.shape[0], SEQ_LEN, hidden_size)
            for block in layers:
                (x,) = block(x)
            out = MagicMock()
            out.last_hidden_state = x
            return out

        # Allow to run the mock blocks to fire the hook and pre-hook calls
        mock_vision_model.side_effect = _forward

        # Ensure robust call
        mock_vision_model.__call__ = _forward

        # Text model
        mock_text_model = MagicMock()
        mock_text_model.eval.return_value = mock_text_model

        # Define pooler_output
        text_out = MagicMock()
        text_out.pooler_output = torch.randn(BATCH_SIZE, hidden_size)
        mock_text_model.return_value = text_out

        return mock_vision_model, mock_text_model, layers

    return _make_vision_language_model_shell


# -----------------------------------------------------------------------------
# Dataset-specific fixtures
# ------------------------------------------------------------------------------
@pytest.fixture
def mock_clip_model(make_mock_vision_language_model: Callable[..., tuple[MagicMock, MagicMock, nn.ModuleList]]) -> Clip:
    """
    A CLIP instance whose HuggingFace backbone is fully mocked.
    Patches CLIPVisionModel.from_pretrained, CLIPTextModel.from_pretrained,
    AutoTokenizer and AutoProcessor so no network call is made.
    """

    blocks = [MockClipBlock() for _ in range(N_LAYERS)]
    mock_vision_hf, mock_text_hf, layers = make_mock_vision_language_model(blocks)

    # Clip reads vision_model.vision_model.encoder.layers from self.blocks
    mock_vision_hf.vision_model = MagicMock()
    mock_vision_hf.vision_model.encoder = MagicMock()
    mock_vision_hf.vision_model.encoder.layers = layers

    # Image processor
    mock_processor = MagicMock()
    mock_processor.return_value = {"pixel_values": torch.randn(BATCH_SIZE, 3, 224, 224)}

    # Tokenizer as a BatchEncoding mock
    mock_tokenizer = MagicMock()

    # This represents the data that the text model needs
    encoding_data = {
        "input_ids": torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.long),
        "attention_mask": torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long),
    }
    mock_batch_encoding = MagicMock()
    mock_batch_encoding.to.return_value = mock_batch_encoding
    mock_batch_encoding.keys.return_value = encoding_data.keys()
    mock_batch_encoding.items.return_value = encoding_data.items()
    mock_batch_encoding.__getitem__ = lambda self, key: encoding_data[key]
    mock_tokenizer.return_value = mock_batch_encoding

    # Setup mock tokenizer with standard special tokens
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.bos_token_id = 1
    mock_tokenizer.eos_token_id = 2

    with (
        patch("waldo.models.multimodal.clip.CLIPVisionModel.from_pretrained", return_value=mock_vision_hf),
        patch("waldo.models.multimodal.clip.CLIPTextModel.from_pretrained", return_value=mock_text_hf),
        patch("waldo.models.multimodal.clip.AutoProcessor.from_pretrained", return_value=mock_processor),
        patch("waldo.models.multimodal.clip.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
    ):
        config = ClipConfig(model_size="base", device="cpu")
        model = Clip(config)

    model.vision_model = mock_vision_hf
    model.text_model = mock_text_hf
    return model


@pytest.fixture
def mock_siglip2_model(
    make_mock_vision_language_model: Callable[..., tuple[nn.Module, nn.Module, nn.ModuleList]],
) -> Siglip2:
    """
    A SigLIP2 instance whose HuggingFace backbone is fully mocked.
    Patches SigLIP2VisionModel.from_pretrained, SigLIP2TextModel.from_pretrained,
    SigLIP2VisionModel.from_pretrained, SigLIP2TextModel.from_pretrained,
    AutoTokenizer and AutoProcessor so no network call is made.
    """

    blocks = [MockSiglip2Block() for _ in range(N_LAYERS)]
    mock_vision_hf, mock_text_hf, layers = make_mock_vision_language_model(blocks)

    # Siglip2 reads vision_model.vision_model.encoder.layers from self.blocks
    mock_vision_hf.vision_model = MagicMock()
    mock_vision_hf.vision_model.encoder = MagicMock()
    mock_vision_hf.vision_model.encoder.layers = layers

    # Image processor
    mock_processor = MagicMock()
    mock_processor.return_value = {"pixel_values": torch.randn(BATCH_SIZE, 3, 224, 224)}

    # Tokenizer as a BatchEncoding mock
    encoding_data = {
        "input_ids": torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.long),
        "attention_mask": torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long),
    }
    mock_batch_encoding = MagicMock()
    mock_batch_encoding.to.return_value = mock_batch_encoding
    mock_batch_encoding.keys.return_value = encoding_data.keys()
    mock_batch_encoding.items.return_value = encoding_data.items()
    mock_batch_encoding.__getitem__ = lambda self, key: encoding_data[key]
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = mock_batch_encoding

    # Setup mock tokenizer with standard special tokens
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.bos_token_id = 1
    mock_tokenizer.eos_token_id = 2

    with (
        patch("waldo.models.multimodal.siglip2.Siglip2VisionModel.from_pretrained", return_value=mock_vision_hf),
        patch("waldo.models.multimodal.siglip2.SiglipVisionModel.from_pretrained", return_value=mock_vision_hf),
        patch("waldo.models.multimodal.siglip2.Siglip2TextModel.from_pretrained", return_value=mock_text_hf),
        patch("waldo.models.multimodal.siglip2.SiglipTextModel.from_pretrained", return_value=mock_text_hf),
        patch("waldo.models.multimodal.siglip2.AutoProcessor.from_pretrained", return_value=mock_processor),
        patch("waldo.models.multimodal.siglip2.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
    ):
        config = Siglip2Config(model_size="base", device="cpu")
        model = Siglip2(config)

    model.vision_model = mock_vision_hf
    model.text_model = mock_text_hf
    return model
