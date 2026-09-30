r"""
DomainNet evaluation entry point unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apps.vision import eval_domainnet as app
from waldo.data import DomainNetDatasetConfig

from ..conftest import CONFIG_DIR, check_model_config, load_config

CONFIG_PATH = CONFIG_DIR / "vision/configs/domainnet.yaml"


# -------------------------------------------------------------------
# Configuration file
# -------------------------------------------------------------------
def test_configuration_file() -> None:
    r"""The configuration file should only contain fields of the app, with valid model and domain values."""
    config = load_config(app.EvaluationConfig, CONFIG_PATH)
    check_model_config(config, ["model_size", "patch_size", "resolution", "pooling"])
    for domain in config.domains:
        DomainNetDatasetConfig(domain=domain)


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def test_eval_trains_on_real_and_tests_on_each_domain(tmp_path: Path) -> None:
    r"""The probes should be trained on the real domain and tested on each other domain independently."""
    build_loader = MagicMock(
        side_effect=lambda config, return_n_classes=False: (config, 345) if return_n_classes else config
    )
    config = load_config(app.EvaluationConfig, CONFIG_PATH, device="cpu", domains=["sketch", "quickdraw"])
    mocks = {"build_model": MagicMock(), "build_loader": build_loader, "build_linear_probing": MagicMock()}
    mocks["build_model"].return_value.model_name = "google/vit-base-patch16-224-in21k"
    with patch.object(app, "RUN_DIR", tmp_path), patch.multiple(app, **mocks):
        app.eval(config)
    kwargs = mocks["build_linear_probing"].return_value.evaluate.call_args.kwargs
    assert (kwargs["train_loader"]["dataset_name"], kwargs["train_loader"]["mode"]) == ("domainnet-real", "train")
    assert list(kwargs["test_loader"]) == ["domainnet-sketch", "domainnet-quickdraw"]
    assert kwargs["cache_path"] == tmp_path / "domainnet" / "google/vit-base-patch16-224-in21k"


# -------------------------------------------------------------------
# Command line interface
# -------------------------------------------------------------------
def test_main_merges_configuration_file_and_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Command line arguments should override the configuration file, which overrides the defaults."""
    monkeypatch.setattr("sys.argv", ["eval", f"config={CONFIG_PATH}", "domains=[sketch]"])
    with patch.object(app, "setup_logging"), patch.object(app, "eval") as run:
        app.main()
    assert run.call_args.kwargs["config"].domains == ["sketch"]
