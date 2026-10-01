r"""
Vision evaluation entry point unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apps.vision import eval as app
from waldo.data import build_transform

from ..conftest import CONFIG_DIR, check_model_config, load_config

CONFIGS = ["vit.yaml", "dinov2.yaml", "dinov3.yaml", "aimv2.yaml"]


def _run(tmp_path: Path, **overrides: object) -> dict[str, MagicMock]:
    r"""Run eval with mocked model, data and linear probing, and return the mocks."""
    model = MagicMock(model_name="facebook/dinov2-base")
    mocks = {
        "build_model": MagicMock(return_value=model),
        "build_loader": MagicMock(
            side_effect=lambda config, return_n_classes=False: ("loader", 10) if return_n_classes else "loader"
        ),
        "build_linear_probing": MagicMock(),
    }
    config = load_config(app.EvaluationConfig, CONFIG_DIR / "vision/configs/dinov2.yaml", device="cpu", **overrides)
    with patch.multiple(app, SAVING_DIR=tmp_path / "savings", **mocks):
        app.eval(config)
    return mocks


# -------------------------------------------------------------------
# Configuration files
# -------------------------------------------------------------------
@pytest.mark.parametrize("name", CONFIGS)
def test_configuration_files(name: str) -> None:
    r"""Each configuration file should only contain fields of the app, with valid model values and no overwrite."""
    config = load_config(app.EvaluationConfig, CONFIG_DIR / "vision/configs" / name)
    assert config.overwrite is False
    check_model_config(config, ["model_size", "patch_size", "resolution", "pooling"])
    build_transform(resolution=config.resolution, transform_type="static")


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
class TestEval:
    r"""Tests for the wiring of the model, data and linear probing."""

    def test_model_and_loaders(self, tmp_path: Path) -> None:
        r"""The model and the static train and test loaders of the dataset should be built from the config."""
        mocks = _run(tmp_path, pooling="all")
        model_config = mocks["build_model"].call_args.kwargs["config"]
        assert (model_config["model_name"], model_config["pooling"], model_config["device"]) == ("dinov2", "all", "cpu")
        loader_configs = [call.kwargs["config"] for call in mocks["build_loader"].call_args_list]
        assert [c["mode"] for c in loader_configs] == ["train", "test"]
        assert all(c["dataset_name"] == "cifar10" and c["transform_type"] == "static" for c in loader_configs)

    def test_config_is_saved(self, tmp_path: Path) -> None:
        r"""The evaluation config should be saved in the run directory of the dataset and model."""
        _run(tmp_path)
        saved = json.loads((tmp_path / "savings/runs/cifar10/facebook/dinov2-base/config.json").read_text())
        assert (saved["model_name"], saved["dataset_name"], saved["pooling"]) == ("dinov2", "cifar10", "content")

    def test_no_overwrite_by_default(self, tmp_path: Path) -> None:
        r"""By default, existing results should never be overwritten."""
        assert app.EvaluationConfig().overwrite is False
        assert _run(tmp_path)["build_linear_probing"].return_value.evaluate.call_args.kwargs["overwrite"] is False

    def test_all_layers_and_modules_by_default(self, tmp_path: Path) -> None:
        r"""By default, all layers and modules should be probed."""
        evaluate = _run(tmp_path)["build_linear_probing"].return_value.evaluate
        assert (evaluate.call_args.kwargs["layers"], evaluate.call_args.kwargs["modules"]) == (None, None)

    def test_selected_layers_and_modules(self, tmp_path: Path) -> None:
        r"""Selected layers and modules should be passed to the linear probing."""
        evaluate = _run(tmp_path, layers=[-1], modules=["rc2norm"])["build_linear_probing"].return_value.evaluate
        assert (evaluate.call_args.kwargs["layers"], evaluate.call_args.kwargs["modules"]) == ([-1], ["rc2norm"])


# -------------------------------------------------------------------
# Run directory
# -------------------------------------------------------------------
class TestLogDir:
    r"""Tests for the directory where the runs are saved."""

    def test_default_log_dir(self, tmp_path: Path) -> None:
        r"""By default, runs should be saved in savings/runs, as for the results of the paper."""
        assert app.EvaluationConfig().log_dir == "runs"
        evaluate = _run(tmp_path)["build_linear_probing"].return_value.evaluate
        assert evaluate.call_args.kwargs["cache_path"] == tmp_path / "savings/runs/cifar10/facebook/dinov2-base"

    def test_custom_log_dir(self, tmp_path: Path) -> None:
        r"""A given log_dir should be used relative to savings/, and nothing should be written in savings/runs."""
        evaluate = _run(tmp_path, log_dir="ablations/final_norm")["build_linear_probing"].return_value.evaluate
        cache_path = tmp_path / "savings/ablations/final_norm/cifar10/facebook/dinov2-base"
        assert evaluate.call_args.kwargs["cache_path"] == cache_path
        assert json.loads((cache_path / "config.json").read_text())["log_dir"] == "ablations/final_norm"
        assert not (tmp_path / "savings/runs").exists()


# -------------------------------------------------------------------
# Command line interface
# -------------------------------------------------------------------
def test_main_merges_configuration_file_and_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Command line arguments should override the configuration file, which overrides the defaults."""
    config_path = CONFIG_DIR / "vision/configs/vit.yaml"
    argv = ["eval", f"config={config_path}", "device=cpu", "modules=[rc2norm]", "log_dir=ablations/final_norm"]
    monkeypatch.setattr("sys.argv", argv)
    with patch.object(app, "setup_logging"), patch.object(app, "eval") as run:
        app.main()
    config = run.call_args.kwargs["config"]
    assert (config.model_name, config.device, config.modules, config.layers) == ("vit", "cpu", ["rc2norm"], ["all"])
    assert config.log_dir == "ablations/final_norm"
