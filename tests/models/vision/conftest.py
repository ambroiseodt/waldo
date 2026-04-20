r"""
Shared fixtures for vision foundation model unit tests.

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

from waldo.models import (
    Dinov3,
    Dinov3Config,
    ViT,
    ViTConfig,
)
from waldo.models.vision.base import BaseVisionModel

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


class MokeViTIntermediate(nn.Module):
    r"""A mock ViT intemediate."""

    def __init__(self):
        super().__init__()
        self.dense = MockLinear()
        self.intermediate_act_fn = MockActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(x))


class MokeViTOutput(nn.Module):
    r"""A mock ViT output layer."""

    def __init__(self):
        super().__init__()
        self.dense = MockLinear()

    def forward(self, x: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.dense(x) + input_tensor


class MockFFN(nn.Module):
    r"""A mock FFN with non-gated activation."""

    def __init__(self):
        super().__init__()
        self.up_proj = MockLinear()
        self.act_fn = MockActivation()
        self.down_proj = MockLinear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class MockGatedFFN(nn.Module):
    r"""A mock MLFFNP with gated activation."""

    def __init__(self):
        super().__init__()
        self.up_proj = MockLinear()
        self.act_fn = MockActivation()
        self.down_proj = MockLinear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class MockViTBlock(nn.Module):
    r"""A mock ViT transformer block."""

    def __init__(self):
        super().__init__()
        self.layernorm_before = MockLayerNorm()
        self.attention = MockAttention()
        self.layernorm_after = MockLayerNorm()
        self.intermediate = MokeViTIntermediate()
        self.output = MokeViTOutput()

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        attn_out, _ = self.attention(self.layernorm_before(x))
        rc1 = x + attn_out
        ffn_in = self.layernorm_after(rc1)
        ffn_out = self.output(self.intermediate(ffn_in), rc1)
        return (ffn_out,)


class MockDinov3Block(nn.Module):
    r"""A mock DINOv3 transformer block."""

    def __init__(self, gated: bool = False):
        super().__init__()
        self.norm1 = MockLayerNorm()
        self.attention = MockAttention()
        self.norm2 = MockLayerNorm()
        self.mlp: nn.Module = MockGatedFFN() if gated else MockFFN()

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        attn_out, _ = self.attention(self.norm1(x))
        rc1 = x + attn_out
        ffn_out = rc1 + self.mlp(self.norm2(rc1))
        return (ffn_out,)


# ---------------------------------------------------------------------------
# Global fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pixel_values() -> torch.Tensor:
    r"""A random batch of pixel values representative of a real dataset sample."""
    return torch.randn(BATCH_SIZE, 3, 224, 224)


@pytest.fixture(scope="session")
def make_mock_hf_model() -> Callable[..., tuple[MagicMock, nn.ModuleList]]:
    r"""A mock model for HuggingFace pretrained model inheriting from BaseVisionModel."""

    def _make_hf_model_shell(
        blocks: list[nn.Module], hidden_size: int = HIDDEN_SIZE
    ) -> tuple[MagicMock, nn.ModuleList]:

        layers = nn.ModuleList(blocks)

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = hidden_size
        mock_model.eval.return_value = mock_model
        mock_model.parameters.return_value = iter([])

        def _forward(pixel_values: torch.Tensor, **kwargs) -> MagicMock:
            x = torch.zeros(pixel_values.shape[0], SEQ_LEN, hidden_size)
            for block in layers:
                (x,) = block(x)
            out = MagicMock()
            out.last_hidden_state = x
            return out

        # Allow to run the mock blocks to fire the hook and pre-hook calls
        mock_model.side_effect = _forward

        # Ensure robust call
        mock_model.__call__ = _forward

        return mock_model, layers

    return _make_hf_model_shell


# -----------------------------------------------------------------------------
# Dataset-specific fixtures
# ------------------------------------------------------------------------------
@pytest.fixture
def mock_vit_model(make_mock_hf_model: Callable[..., tuple[MagicMock, nn.ModuleList]]) -> ViT:
    """
    A ViT instance whose HuggingFace backbone is fully mocked.
    Patches ViTModel.from_pretrained so no network call is made.
    Returns the ViT wrapper (waldo.models.ViT).
    """

    blocks = [MockViTBlock() for _ in range(N_LAYERS)]
    mock_hf, layers = make_mock_hf_model(blocks)

    # ViT reads encoder.layer for self.blocks
    mock_hf.encoder = MagicMock()
    mock_hf.encoder.layer = layers

    with patch("waldo.models.vision.vit.ViTModel.from_pretrained", return_value=mock_hf):
        config = ViTConfig(model_size="base", patch_size=16, device="cpu")
        model = ViT(config)

    model.model = mock_hf
    return model


@pytest.fixture
def mock_dinov3_model(make_mock_hf_model: Callable[..., tuple[MagicMock, nn.ModuleList]]) -> BaseVisionModel:
    """
    A Dinov3 instance backed by standard MLP blocks (no SwiGLU).
    Patches DINOv3ViTModel.from_pretrained.
    Returns the Dinov3 wrapper (waldo.models.Dinov3).
    """

    blocks = [MockDinov3Block(gated=False) for _ in range(N_LAYERS)]
    mock_hf, layers = make_mock_hf_model(blocks)
    mock_hf.layer = layers

    with patch("waldo.models.vision.dinov3.DINOv3ViTModel.from_pretrained", return_value=mock_hf):
        config = Dinov3Config(model_size="vitb16", device="cpu")
        model = Dinov3(config)

    model.model = mock_hf
    return model


@pytest.fixture
def mock_dinov3_gated_model(make_mock_hf_model: Callable[..., tuple[MagicMock, nn.ModuleList]]) -> BaseVisionModel:
    """
    A Dinov3 instance backed by GatedMLP (SwiGLU) blocks.
    Patches DINOv3ViTModel.from_pretrained.
    Returns the Dinov3 wrapper (waldo.models.Dinov3).
    """

    blocks = [MockDinov3Block(gated=True) for _ in range(N_LAYERS)]
    mock_hf, layers = make_mock_hf_model(blocks)
    mock_hf.layer = layers

    with patch("waldo.models.vision.dinov3.DINOv3ViTModel.from_pretrained", return_value=mock_hf):
        config = Dinov3Config(model_size="vit7b16", device="cpu")
        model = Dinov3(config)

    model.model = mock_hf
    return model
