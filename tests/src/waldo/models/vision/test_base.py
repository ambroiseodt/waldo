r"""
Base vision model unit tests, run on every architecture.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import pytest
import torch

from waldo.models import TRANSFORMER_MODULES, BaseVisionModel, mean_pooling
from waldo.models.transformer_block import RC2NORM

from ..conftest import assert_block_identities, capture_block_inputs
from .conftest import BATCH_SIZE, HIDDEN_SIZE, N_LAYERS, N_PATCHES, VISION_SPECS, MakeVisionModel


def _full_sequences(model: BaseVisionModel) -> None:
    r"""Disable the pooling so that get_embeddings returns the full sequences of tokens."""
    model.pooling = lambda hidden_states: hidden_states


# -------------------------------------------------------------------
# Embeddings
# -------------------------------------------------------------------
class TestGetEmbeddings:
    r"""Tests for the representations recovered in a single forward pass."""

    def test_default_keys(self, make_vision_model: MakeVisionModel, vision_spec: str, images: torch.Tensor) -> None:
        r"""By default, all layers and all modules of TRANSFORMER_MODULES should be recovered."""
        model, _ = make_vision_model(vision_spec)
        embeddings = model.get_embeddings(images)
        assert set(embeddings) == {f"block_{i}_{m}" for i in range(N_LAYERS) for m in TRANSFORMER_MODULES}

    def test_embeddings_are_pooled_on_cpu(
        self, make_vision_model: MakeVisionModel, vision_spec: str, images: torch.Tensor
    ) -> None:
        r"""Each embedding should be a pooled (N, E) tensor on CPU, or (N, 4E) for fc1 and act."""
        model, _ = make_vision_model(vision_spec)
        for key, embedding in model.get_embeddings(images).items():
            assert embedding.device.type == "cpu"
            assert embedding.shape[0] == BATCH_SIZE
            assert embedding.ndim == 2, key

    def test_block_identities(self, make_vision_model: MakeVisionModel, vision_spec: str, images: torch.Tensor) -> None:
        r"""The 8 recovered modules of every block should be consistent with the block computations."""
        model, _ = make_vision_model(vision_spec)
        _full_sequences(model)
        with torch.inference_mode(), capture_block_inputs(model.blocks) as inputs:
            embeddings = model.get_embeddings(images)
            for idx, block in enumerate(model.blocks):
                adapter = model.block_adapter(block)
                outputs = {m: embeddings[f"block_{idx}_{m}"] for m in TRANSFORMER_MODULES}
                assert_block_identities(block, adapter, inputs[idx], outputs)

    def test_last_rc2_is_last_block_output(
        self, make_vision_model: MakeVisionModel, vision_spec: str, images: torch.Tensor
    ) -> None:
        r"""rc2 of the last layer should be the output of the last block, before the final norm."""
        model, hf_model = make_vision_model(vision_spec)
        _full_sequences(model)
        rc2 = model.get_embeddings(images, layers=[-1], modules=["rc2"])[f"block_{N_LAYERS - 1}_rc2"]
        with torch.inference_mode():
            torch.testing.assert_close(model.final_norm(rc2), hf_model(pixel_values=images).last_hidden_state)

    def test_rc2norm_is_standard_output(
        self, make_vision_model: MakeVisionModel, vision_spec: str, images: torch.Tensor
    ) -> None:
        r"""rc2norm should be the pooled output of the final norm, i.e. the last_hidden_state of HuggingFace."""
        model, hf_model = make_vision_model(vision_spec)
        embedding = model.get_embeddings(images, layers=[-1], modules=[RC2NORM])[f"block_{N_LAYERS - 1}_{RC2NORM}"]
        with torch.inference_mode():
            last_hidden_state = hf_model(pixel_values=images).last_hidden_state
        expected = mean_pooling(last_hidden_state, n_prefix_tokens=VISION_SPECS[vision_spec].n_prefix_tokens)
        torch.testing.assert_close(embedding, expected)

    def test_hooks_are_removed(
        self, make_vision_model: MakeVisionModel, vision_spec: str, images: torch.Tensor
    ) -> None:
        r"""No hook should remain on the model after get_embeddings, even if the forward pass fails."""
        model, _ = make_vision_model(vision_spec)
        model.get_embeddings(images, modules=TRANSFORMER_MODULES + [RC2NORM])
        with pytest.raises((RuntimeError, ValueError)):
            model.get_embeddings(torch.randn(BATCH_SIZE, 5, 3, 3))
        for module in model.model.modules():
            assert not module._forward_hooks and not module._forward_pre_hooks


# -------------------------------------------------------------------
# Layer and module selection
# -------------------------------------------------------------------
class TestSelection:
    r"""Tests for the selection of layers and modules."""

    def test_selected_keys(self, make_vision_model: MakeVisionModel, images: torch.Tensor) -> None:
        r"""Only the requested layers and modules should be recovered, with nonnegative layer indices."""
        model, _ = make_vision_model("vit")
        embeddings = model.get_embeddings(images, layers=[0, -1], modules=["act", RC2NORM])
        assert set(embeddings) == {"block_0_act", f"block_{N_LAYERS - 1}_act", f"block_{N_LAYERS - 1}_{RC2NORM}"}

    def test_selection_matches_full_extraction(self, make_vision_model: MakeVisionModel, images: torch.Tensor) -> None:
        r"""Selected representations should be identical to the ones of a full extraction."""
        model, _ = make_vision_model("vit")
        full = model.get_embeddings(images)
        selected = model.get_embeddings(images, layers=[1], modules=["ln2"])
        torch.testing.assert_close(selected["block_1_ln2"], full["block_1_ln2"])

    @pytest.mark.parametrize("layers", [[N_LAYERS], [-N_LAYERS - 1]])
    def test_out_of_range_layer_raises(
        self, make_vision_model: MakeVisionModel, images: torch.Tensor, layers: list[int]
    ) -> None:
        r"""Out of range layer indices should raise a ValueError."""
        model, _ = make_vision_model("vit")
        with pytest.raises(ValueError, match="out of range"):
            model.get_embeddings(images, layers=layers)

    def test_invalid_module_raises(self, make_vision_model: MakeVisionModel, images: torch.Tensor) -> None:
        r"""Unknown module names should raise a ValueError."""
        model, _ = make_vision_model("vit")
        with pytest.raises(ValueError, match="Invalid modules"):
            model.get_embeddings(images, modules=["unexisting"])

    def test_rc2norm_without_last_layer_raises(self, make_vision_model: MakeVisionModel, images: torch.Tensor) -> None:
        r"""rc2norm is only defined for the last layer, which must be requested."""
        model, _ = make_vision_model("vit")
        with pytest.raises(ValueError, match="only defined for the last layer"):
            model.get_embeddings(images, layers=[0], modules=[RC2NORM])


# -------------------------------------------------------------------
# Pooling
# -------------------------------------------------------------------
class TestPooling:
    r"""Tests for the pooling over the sequence of tokens."""

    def test_content_pooling_excludes_prefix_tokens(self, make_vision_model: MakeVisionModel, vision_spec: str) -> None:
        r"""Content pooling should average the patch tokens only, excluding the CLS and register tokens."""
        model, _ = make_vision_model(vision_spec)
        n_prefix = VISION_SPECS[vision_spec].n_prefix_tokens
        assert model.n_prefix_tokens == n_prefix
        hidden_states = torch.randn(BATCH_SIZE, n_prefix + N_PATCHES, HIDDEN_SIZE)
        hidden_states[:, :n_prefix] = 1e6
        torch.testing.assert_close(model.pooling(hidden_states), hidden_states[:, n_prefix:].mean(dim=1))

    @pytest.mark.parametrize("name", [name for name, spec in VISION_SPECS.items() if spec.n_prefix_tokens > 0])
    def test_all_pooling_concatenates_cls_and_content(self, make_vision_model: MakeVisionModel, name: str) -> None:
        r"""All pooling should concatenate the CLS token and the mean over the patch tokens."""
        model, _ = make_vision_model(name, pooling="all")
        n_prefix = VISION_SPECS[name].n_prefix_tokens
        hidden_states = torch.randn(BATCH_SIZE, n_prefix + N_PATCHES, HIDDEN_SIZE)
        expected = torch.cat([hidden_states[:, 0], hidden_states[:, n_prefix:].mean(dim=1)], dim=-1)
        torch.testing.assert_close(model.pooling(hidden_states), expected)

    def test_all_pooling_doubles_embedding_dimension(
        self, make_vision_model: MakeVisionModel, images: torch.Tensor
    ) -> None:
        r"""With all pooling, the embeddings of every module should be twice as wide."""
        content = make_vision_model("dinov2")[0].get_embeddings(images, layers=[0])
        pooled = make_vision_model("dinov2", pooling="all")[0].get_embeddings(images, layers=[0])
        for key in content:
            assert pooled[key].shape[1] == 2 * content[key].shape[1]

    def test_invalid_pooling_raises(self, make_vision_model: MakeVisionModel, vision_spec: str) -> None:
        r"""Unknown pooling types should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid pooling"):
            make_vision_model(vision_spec, pooling="cls")


# -------------------------------------------------------------------
# Properties
# -------------------------------------------------------------------
class TestProperties:
    r"""Tests for the properties of the model."""

    def test_model_in_eval_mode(self, make_vision_model: MakeVisionModel, vision_spec: str) -> None:
        r"""The model should be loaded in evaluation mode."""
        model, _ = make_vision_model(vision_spec)
        assert not model.model.training

    def test_n_layers_and_embedding_dim(self, make_vision_model: MakeVisionModel, vision_spec: str) -> None:
        r"""n_layers and embedding_dim should match the configuration of the HuggingFace model."""
        model, _ = make_vision_model(vision_spec)
        assert model.n_layers == N_LAYERS
        assert model.embedding_dim == HIDDEN_SIZE

    def test_gated_ffn_act_is_pre_hook(self, make_vision_model: MakeVisionModel, vision_spec: str) -> None:
        r"""For gated FFNs, act should be recovered as the input of the down projection."""
        model, _ = make_vision_model(vision_spec)
        adapter = model.block_adapter(model.blocks[0])
        assert ("act" in adapter.pre_hooks) == VISION_SPECS[vision_spec].gated
        assert "rc1" in adapter.pre_hooks

    def test_repr_counts_parameters(self, make_vision_model: MakeVisionModel) -> None:
        r"""__repr__ should give the class name and the number of trainable parameters."""
        model, hf_model = make_vision_model("vit")
        n_params = sum(p.numel() for p in hf_model.parameters())
        assert repr(model) == f"ViT model with {n_params:,} trainable parameters."
