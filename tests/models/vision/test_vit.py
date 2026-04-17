r"""
ViT unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import pytest
import torch
from torch.nn import ModuleList

from waldo.models import ViT, ViTBlockAdapter, ViTConfig
from waldo.models.transformer_block import TRANSFORMER_MODULES

from .conftest import (
    BATCH_SIZE,
    HIDDEN_SIZE,
    N_LAYERS,
    SEQ_LEN,
    MockViTBlock,
)


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestViTConfig:
    @pytest.mark.parametrize(
        "size,patch",
        [
            ("base", 16),
            ("base", 32),
            ("large", 16),
            ("large", 32),
            ("huge", 14),
        ],
    )
    def test_valid_size_patch_combinations_accepted(self, size: str, patch: int) -> None:
        cfg = ViTConfig(model_size=size, patch_size=patch)
        assert cfg.model_size == size
        assert cfg.patch_size == patch

    def test_invalid_model_size_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid model size"):
            ViTConfig(model_size="xlarge", patch_size=16)

    def test_invalid_patch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid patch size"):
            ViTConfig(model_size="base", patch_size=8)

    def test_defaults(self) -> None:
        cfg = ViTConfig()
        assert cfg.model_size == "base"
        assert cfg.patch_size == 16
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# BlockAdapter class
# ---------------------------------------------------------------------------


def _make_adapter() -> tuple[ViTBlockAdapter, MockViTBlock]:
    block = MockViTBlock()
    return ViTBlockAdapter(block), block


class TestViTBlockAdapter:
    def test_module_map_has_all_eight_keys(self) -> None:
        adapter, _ = _make_adapter()
        assert set(adapter.module_map().keys()) == set(TRANSFORMER_MODULES)

    def test_rc1_in_pre_hooks(self) -> None:
        adapter, _ = _make_adapter()
        assert "rc1" in adapter.pre_hooks

    def test_only_rc1_in_pre_hooks(self) -> None:
        """Standard ViT FFN has no SwiGLU — only rc1 needs a pre-hook."""
        adapter, _ = _make_adapter()
        assert adapter.pre_hooks == {"rc1"}

    def test_ln1_and_ln2_are_different_modules(self) -> None:
        adapter, block = _make_adapter()
        assert adapter.ln1 is not adapter.ln2
        assert adapter.ln1 is block.layernorm_before
        assert adapter.ln2 is block.layernorm_after

    def test_rc1_and_ln2_share_layernorm_after(self) -> None:
        adapter, block = _make_adapter()
        assert adapter.rc1 is block.layernorm_after
        assert adapter.ln2 is block.layernorm_after

    def test_fc1_points_to_intermediate_dense(self) -> None:
        adapter, block = _make_adapter()
        assert adapter.fc1 is block.intermediate.dense

    def test_fc2_points_to_output_dense(self) -> None:
        adapter, block = _make_adapter()
        assert adapter.fc2 is block.output.dense

    def test_act_points_to_intermediate_act_fn(self) -> None:
        adapter, block = _make_adapter()
        assert adapter.act is block.intermediate.intermediate_act_fn

    def test_rc2_is_block_itself(self) -> None:
        adapter, block = _make_adapter()
        assert adapter.rc2 is block

    def test_mha_points_to_attention(self) -> None:
        adapter, block = _make_adapter()
        
        assert adapter.mha is block.attention


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class TestViTModel:
    def test_invalid_model_name_raises(self, mock_vit_model: ViT) -> None:
        with pytest.raises(ValueError, match="not valid"):
            mock_vit_model._load_model("google/vit-giant-patch16-224-in21k", "cpu")

    def test_model_in_eval_mode_after_loading(self, mock_vit_model: ViT) -> None:
        mock_vit_model.model.eval.assert_called()

    def test_blocks_returns_module_list(self, mock_vit_model: ViT) -> None:
        assert isinstance(mock_vit_model.blocks, ModuleList)

    def test_blocks_length_equals_n_layers(self, mock_vit_model: ViT) -> None:
        assert len(mock_vit_model.blocks) == N_LAYERS

    def test_block_adapter_is_vit_adapter(self, mock_vit_model: ViT) -> None:
        assert mock_vit_model.block_adapter is ViTBlockAdapter

    def test_embedding_dim_equals_hidden_size(self, mock_vit_model: ViT) -> None:
        assert mock_vit_model.embedding_dim == HIDDEN_SIZE

    def test_n_layers_equals_n_layers_constant(self, mock_vit_model: ViT) -> None:
        assert mock_vit_model.n_layers == N_LAYERS

    def test_pooling_returns_n_e_shape(self, mock_vit_model: ViT) -> None:
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mock_vit_model.pooling(hidden)
        assert out.shape == (BATCH_SIZE, HIDDEN_SIZE)

    def test_pooling_result_is_on_cpu(self, mock_vit_model: ViT) -> None:
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mock_vit_model.pooling(hidden)
        assert out.device.type == "cpu"

    def test_pooling_excludes_cls_token(self, mock_vit_model: ViT) -> None:
        """CLS is position 0; setting it to a large value should shift the
        mean-with-CLS away from mean-without-CLS."""
        torch.manual_seed(42)
        hidden = torch.randn(1, SEQ_LEN, HIDDEN_SIZE)
        hidden[:, 0, :] = 100.0
        out_excl = mock_vit_model.pooling(hidden)
        out_incl = hidden.mean(dim=1).cpu()
        assert not torch.allclose(out_excl, out_incl)

    def test_returns_all_keys(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        embeddings = mock_vit_model.get_embeddings(pixel_values)
        expected = {f"block_{layer}_{module}" for layer in range(N_LAYERS) for module in TRANSFORMER_MODULES}
        assert set(embeddings.keys()) == expected

    def test_embedding_shapes_are_n_e(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        embeddings = mock_vit_model.get_embeddings(pixel_values)
        for key, tensor in embeddings.items():
            assert tensor.shape == (BATCH_SIZE, HIDDEN_SIZE), (
                f"{key}: expected ({BATCH_SIZE}, {HIDDEN_SIZE}), got {tensor.shape}"
            )

    def test_embeddings_are_on_cpu(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        embeddings = mock_vit_model.get_embeddings(pixel_values)
        for key, tensor in embeddings.items():
            assert tensor.device.type == "cpu", f"{key} is on {tensor.device}"

    def test_no_extra_keys_produced(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        embeddings = mock_vit_model.get_embeddings(pixel_values)
        for key in embeddings:
            parts = key.split("_")
            assert parts[0] == "block"
            assert int(parts[1]) < N_LAYERS
            assert "_".join(parts[2:]) in TRANSFORMER_MODULES

    def test_layer_filter_single(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        embeddings = mock_vit_model.get_embeddings(pixel_values, layers=[0])
        layers_found = {int(k.split("_")[1]) for k in embeddings}
        assert layers_found == {0}

    def test_layer_filter_subset(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        embeddings = mock_vit_model.get_embeddings(pixel_values, layers=[0, 2])
        layers_found = {int(k.split("_")[1]) for k in embeddings}
        assert layers_found == {0, 2}

    def test_module_filter_single(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        embeddings = mock_vit_model.get_embeddings(pixel_values, modules=["fc1"])
        modules_found = {k.split("_")[2] for k in embeddings}
        assert modules_found == {"fc1"}

    def test_module_filter_rc1_pre_hook(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        """rc1 is a pre-hook module — filtering to it alone must still produce N_LAYERS results."""
        embeddings = mock_vit_model.get_embeddings(pixel_values, modules=["rc1"])
        assert len(embeddings) == N_LAYERS
        assert all(k.endswith("_rc1") for k in embeddings)

    def test_empty_layers_returns_empty(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        assert mock_vit_model.get_embeddings(pixel_values, layers=[]) == {}

    def test_empty_modules_returns_empty(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        assert mock_vit_model.get_embeddings(pixel_values, modules=[]) == {}

    def test_combined_layer_and_module_filter(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        embeddings = mock_vit_model.get_embeddings(pixel_values, layers=[2], modules=["mha", "rc2"])
        assert set(embeddings.keys()) == {"block_2_mha", "block_2_rc2"}
