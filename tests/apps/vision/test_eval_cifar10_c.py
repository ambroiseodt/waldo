r"""
Cifar10-C evaluation entry point unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apps.vision import eval_cifar10_c as app
from waldo.data import Cifar10CDatasetConfig

from ..conftest import CONFIG_DIR, check_model_config, load_config, saving_dirs

CONFIG_PATH = CONFIG_DIR / "vision/configs/cifar10_c.yaml"


# -------------------------------------------------------------------
# Configuration file
# -------------------------------------------------------------------
def test_configuration_file() -> None:
    r"""The configuration file should only contain fields of the app, with valid model and corruption values."""
    config = load_config(app.EvaluationConfig, CONFIG_PATH)
    check_model_config(config, ["model_size", "patch_size", "resolution", "pooling"])
    for corruption in config.corruptions:
        for severity in config.severities:
            Cifar10CDatasetConfig(corruption_type=corruption, corruption_severity=severity)


def _run(tmp_path: Path, **overrides: object) -> dict:
    r"""Run eval with mocked model, data and linear probing, and return the arguments of the evaluation."""
    build_loader = MagicMock(
        side_effect=lambda config, return_n_classes=False: (config, 10) if return_n_classes else config
    )
    config = load_config(
        app.EvaluationConfig, CONFIG_PATH, device="cpu", corruptions=["fog"], severities=[1, 5], **overrides
    )
    mocks = {"build_model": MagicMock(), "build_loader": build_loader, "build_linear_probing": MagicMock()}
    mocks["build_model"].return_value.model_name = "google/vit-base-patch16-224-in21k"
    with saving_dirs(app, tmp_path), patch.multiple(app, **mocks):
        app.eval(config)
    return mocks["build_linear_probing"].return_value.evaluate.call_args.kwargs


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def test_eval_trains_on_cifar10_and_tests_on_each_corruption(tmp_path: Path) -> None:
    r"""The probes should be trained on Cifar10 and tested on each corruption and severity independently."""
    kwargs = _run(tmp_path)
    assert kwargs["train_loader"]["dataset_name"] == "cifar10"
    assert list(kwargs["test_loader"]) == ["cifar10_c-corruption-fog-severity-1", "cifar10_c-corruption-fog-severity-5"]
    assert all(loader["mode"] == "test" for loader in kwargs["test_loader"].values())


# -------------------------------------------------------------------
# Run directory
# -------------------------------------------------------------------
def test_default_log_dir(tmp_path: Path) -> None:
    r"""By default, runs should be saved in RUN_DIR, i.e. savings/runs, as for the results of the paper."""
    assert app.EvaluationConfig().log_dir is None
    assert app.RUN_DIR == app.SAVING_DIR / "runs"
    cache_path = _run(tmp_path)["cache_path"]
    assert cache_path == tmp_path / "savings/runs/cifar10_c/google/vit-base-patch16-224-in21k"
    assert (cache_path / "config.json").exists()


def test_custom_log_dir(tmp_path: Path) -> None:
    r"""A given log_dir should be used relative to savings/, and nothing should be written in RUN_DIR."""
    cache_path = _run(tmp_path, log_dir="runs_v2")["cache_path"]
    assert cache_path == tmp_path / "savings/runs_v2/cifar10_c/google/vit-base-patch16-224-in21k"
    assert not (tmp_path / "savings/runs").exists()


# -------------------------------------------------------------------
# Command line interface
# -------------------------------------------------------------------
def test_main_merges_configuration_file_and_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Command line arguments should override the configuration file, which overrides the defaults."""
    monkeypatch.setattr("sys.argv", ["eval", f"config={CONFIG_PATH}", "severities=[5]", "log_dir=runs_v2"])
    with patch.object(app, "setup_logging"), patch.object(app, "eval") as run:
        app.main()
    config = run.call_args.kwargs["config"]
    assert (config.severities, config.log_dir) == ([5], "runs_v2")
