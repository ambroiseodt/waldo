r"""
Linear probing unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from waldo.eval import LinearProbing, LinearProbingConfig

# Default parameters
N_SAMPLES = 40
N_CLASSES = 4
BATCH_SIZE = 8
KEYS = ["block_0_mha", "block_0_rc2", "block_1_mha", "block_1_rc2"]


class FakeModel(torch.nn.Module):
    r"""Model whose embeddings are the one-hot labels encoded in the inputs, so that linear probes are perfect."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple] = []

    def get_embeddings(
        self, x: torch.Tensor, layers: list[int] | None = None, modules: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        self.calls.append((layers, modules))
        return {key: x.clone() for key in KEYS} | {"not_a_block_key": x.clone()}


def _loader(seed: int) -> DataLoader:
    r"""Inputs are noisy one-hot encodings of the labels."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(N_SAMPLES) % N_CLASSES
    x = torch.nn.functional.one_hot(labels, N_CLASSES).float() + 0.01 * torch.randn(
        N_SAMPLES, N_CLASSES, generator=generator
    )
    return DataLoader(TensorDataset(x, labels), batch_size=BATCH_SIZE)


@pytest.fixture
def probing() -> LinearProbing:
    return LinearProbing(LinearProbingConfig(device="cpu", max_iter=200))


@pytest.fixture(autouse=True)
def no_garbage_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Skip the garbage collections run after each batch and probe, which only free memory and are slow in tests."""
    monkeypatch.setattr("waldo.eval.linear.gc.collect", lambda: 0)


def _results(path: Path) -> dict[str, float]:
    return {f.name: json.loads(f.read_text())["acc_test"] for f in sorted(path.glob("*.json"))}


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestLinearProbingConfig:
    r"""Tests for config class valid entries and default values."""

    def test_defaults(self) -> None:
        r"""Default config should follow the DINOv2 linear probing setup."""
        config = LinearProbingConfig()
        assert (config.tol, config.max_iter, config.seed) == (1e-4, 1000, 42)

    @pytest.mark.parametrize(
        "kwargs, match", [({"tol": -1.0}, "Invalid tolerance"), ({"max_iter": 0}, "Invalid maximum")]
    )
    def test_invalid_values_raise(self, kwargs: dict, match: str) -> None:
        r"""Negative tolerances and nonpositive numbers of iterations should raise a ValueError."""
        with pytest.raises(ValueError, match=match):
            LinearProbingConfig(**kwargs)


# -------------------------------------------------------------------
# Classifier
# -------------------------------------------------------------------
class TestPipeline:
    r"""Tests for the choice of the linear classifier."""

    def test_sklearn_on_cpu(self, probing: LinearProbing) -> None:
        r"""On CPU, the classifier should be a standardized L2 logistic regression from sklearn."""
        scaler, solver = (step for _, step in probing._pipeline().steps)
        assert isinstance(scaler, StandardScaler)
        assert isinstance(solver, LogisticRegression)
        assert (solver.C, solver.l1_ratio, solver.tol, solver.max_iter) == (1.0, 0.0, 1e-4, 200)

    def test_cuml_on_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        r"""On GPU, the classifier should be the cuML logistic regression, imported only then."""

        class Estimator:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def fit(self) -> None: ...

            def transform(self) -> None: ...

        for name in ["cuml", "cuml.linear_model", "cuml.preprocessing"]:
            module = types.ModuleType(name)
            module.LogisticRegression = module.StandardScaler = Estimator
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        _, solver = (step for _, step in LinearProbing(LinearProbingConfig(device="cuda:0"))._pipeline().steps)
        assert isinstance(solver, Estimator)
        assert solver.kwargs["penalty"] == "l2" and solver.kwargs["C"] == 1.0


# -------------------------------------------------------------------
# Evaluation on a single test set
# -------------------------------------------------------------------
class TestEvaluateSingle:
    r"""Tests for the evaluation on a pair of train and test loaders."""

    def test_results(self, probing: LinearProbing, tmp_path: Path) -> None:
        r"""Each layer and module should get an accuracy file, and the temporary embeddings should be deleted."""
        probing.evaluate(FakeModel(), _loader(0), _loader(1), cache_path=tmp_path, overwrite=False)
        results = _results(tmp_path / "results")
        assert set(results) == {f"layer_{k.split('_')[1]}_module_{k.split('_')[2]}.json" for k in KEYS}
        assert all(acc == 1.0 for acc in results.values())
        assert not (tmp_path / "results" / "temp_embeddings").exists()
        assert set(json.loads((tmp_path / "info.json").read_text())) == {
            "train_extraction_time",
            "test_extraction_time",
            "eval_time",
        }

    def test_layers_and_modules_are_forwarded(self, probing: LinearProbing, tmp_path: Path) -> None:
        r"""The selected layers and modules should be passed to get_embeddings for every batch."""
        model = FakeModel()
        probing.evaluate(model, _loader(0), _loader(1), tmp_path, overwrite=False, layers=[-1], modules=["rc2norm"])
        assert model.calls and all(call == ([-1], ["rc2norm"]) for call in model.calls)

    def test_complete_results_are_skipped(self, probing: LinearProbing, tmp_path: Path) -> None:
        r"""Without overwrite, a complete evaluation (with info.json) should not be run again."""
        probing.evaluate(FakeModel(), _loader(0), _loader(1), tmp_path, overwrite=False)
        model = FakeModel()
        probing.evaluate(model, _loader(0), _loader(1), tmp_path, overwrite=False)
        assert model.calls == []

    def test_interrupted_results_are_recomputed(self, probing: LinearProbing, tmp_path: Path) -> None:
        r"""Results without info.json come from an interrupted run and should be recomputed."""
        probing.evaluate(FakeModel(), _loader(0), _loader(1), tmp_path, overwrite=False)
        (tmp_path / "info.json").unlink()
        (tmp_path / "results" / "layer_0_module_mha.json").unlink()
        model = FakeModel()
        probing.evaluate(model, _loader(0), _loader(1), tmp_path, overwrite=False)
        assert model.calls
        assert len(_results(tmp_path / "results")) == len(KEYS)
        assert (tmp_path / "info.json").exists()

    def test_overwrite_recomputes(self, probing: LinearProbing, tmp_path: Path) -> None:
        r"""With overwrite, a complete evaluation should be run again."""
        probing.evaluate(FakeModel(), _loader(0), _loader(1), tmp_path, overwrite=False)
        model = FakeModel()
        probing.evaluate(model, _loader(0), _loader(1), tmp_path, overwrite=True)
        assert model.calls


# -------------------------------------------------------------------
# Evaluation on several test sets
# -------------------------------------------------------------------
class TestEvaluateMultiple:
    r"""Tests for the evaluation on a train loader and several test loaders."""

    def test_results_per_test_set(self, probing: LinearProbing, tmp_path: Path) -> None:
        r"""Each test set should get its own results folder, and info.json should list the evaluated test sets."""
        probing.evaluate(FakeModel(), _loader(0), {"a": _loader(1), "b": _loader(2)}, tmp_path, overwrite=False)
        for key in ["a", "b"]:
            results = _results(tmp_path / "results" / key)
            assert len(results) == len(KEYS) and all(acc == 1.0 for acc in results.values())
        assert json.loads((tmp_path / "info.json").read_text())["test_sets"] == ["a", "b"]

    def test_only_missing_test_sets_are_evaluated(self, probing: LinearProbing, tmp_path: Path) -> None:
        r"""Test sets already evaluated should be neither recomputed nor overwritten."""
        probing.evaluate(FakeModel(), _loader(0), {"a": _loader(1)}, tmp_path, overwrite=False)
        result_file = tmp_path / "results" / "a" / "layer_0_module_mha.json"
        mtime = result_file.stat().st_mtime_ns
        model = FakeModel()
        probing.evaluate(model, _loader(0), {"a": _loader(1), "b": _loader(2)}, tmp_path, overwrite=False)
        n_batches = N_SAMPLES // BATCH_SIZE
        assert len(model.calls) == 2 * n_batches
        assert result_file.stat().st_mtime_ns == mtime
        assert json.loads((tmp_path / "info.json").read_text())["test_sets"] == ["a", "b"]

    def test_complete_results_are_skipped(self, probing: LinearProbing, tmp_path: Path) -> None:
        r"""When all test sets are evaluated, nothing should be run again."""
        probing.evaluate(FakeModel(), _loader(0), {"a": _loader(1)}, tmp_path, overwrite=False)
        model = FakeModel()
        probing.evaluate(model, _loader(0), {"a": _loader(1)}, tmp_path, overwrite=False)
        assert model.calls == []

    def test_results_without_test_sets_in_info_are_complete(self, probing: LinearProbing, tmp_path: Path) -> None:
        r"""Runs saved before test sets were listed in info.json should be complete for their existing results."""
        probing.evaluate(FakeModel(), _loader(0), {"a": _loader(1)}, tmp_path, overwrite=False)
        info_path = tmp_path / "info.json"
        info = json.loads(info_path.read_text())
        del info["test_sets"]
        info_path.write_text(json.dumps(info))
        model = FakeModel()
        probing.evaluate(model, _loader(0), {"a": _loader(1)}, tmp_path, overwrite=False)
        assert model.calls == []

    def test_interrupted_test_set_is_recomputed(self, probing: LinearProbing, tmp_path: Path) -> None:
        r"""A test set with results but not listed in info.json should be recomputed."""
        probing.evaluate(FakeModel(), _loader(0), {"a": _loader(1)}, tmp_path, overwrite=False)
        partial = tmp_path / "results" / "b"
        partial.mkdir()
        (partial / "layer_0_module_mha.json").write_text(json.dumps({"acc_test": 0.0}))
        probing.evaluate(FakeModel(), _loader(0), {"a": _loader(1), "b": _loader(2)}, tmp_path, overwrite=False)
        results = _results(partial)
        assert len(results) == len(KEYS) and all(acc == 1.0 for acc in results.values())

    def test_overwrite_recomputes(self, probing: LinearProbing, tmp_path: Path) -> None:
        r"""With overwrite, all test sets should be evaluated again."""
        probing.evaluate(FakeModel(), _loader(0), {"a": _loader(1)}, tmp_path, overwrite=False)
        model = FakeModel()
        probing.evaluate(model, _loader(0), {"a": _loader(1)}, tmp_path, overwrite=True)
        assert model.calls
        assert np.isclose(_results(tmp_path / "results" / "a")["layer_0_module_mha.json"], 1.0)
