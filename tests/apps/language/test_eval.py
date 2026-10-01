r"""
Language evaluation entry point unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from apps.language import eval as app
from waldo.data import MTEBConfig
from waldo.models import TRANSFORMER_MODULES

from ..conftest import CONFIG_DIR, check_model_config, load_config

CONFIGS = ["gemma3.yaml", "llama3.yaml", "qwen3.yaml"]


def _run(tmp_path: Path, **overrides: object) -> tuple[MagicMock, MagicMock]:
    r"""Run eval with mocked model and benchmark, and return both."""
    model = MagicMock(model_name="Qwen/Qwen3-0.6B", n_layers=3)
    benchmark = MagicMock(n_tasks=2, domains={"News"}, languages={"eng"})
    config = load_config(app.EvaluationConfig, CONFIG_DIR / "language/configs/qwen3.yaml", device="cpu", **overrides)
    with (
        patch.object(app, "SAVING_DIR", tmp_path / "savings"),
        patch.object(app, "build_model", return_value=model),
        patch.object(app, "build_mteb", return_value=benchmark),
    ):
        app.eval(config)
    return model, benchmark


# -------------------------------------------------------------------
# Configuration files
# -------------------------------------------------------------------
@pytest.mark.parametrize("name", CONFIGS)
def test_configuration_files(name: str) -> None:
    r"""Each configuration file should only contain fields of the app, with valid model and benchmark values."""
    config = load_config(app.EvaluationConfig, CONFIG_DIR / "language/configs" / name)
    check_model_config(config, ["model_size"])
    MTEBConfig(benchmark_name=config.benchmark_name, task_type=config.task_type, max_n_docs=config.max_n_docs)


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
class TestEval:
    r"""Tests for the loop over the probed layers and modules."""

    def test_all_layers_and_modules(self, tmp_path: Path) -> None:
        r"""With all layers and modules, each pair should be probed and evaluated once."""
        model, benchmark = _run(tmp_path, layers=["all"], modules=["all"])
        expected = [call(layer=layer, module=module) for layer in range(3) for module in TRANSFORMER_MODULES]
        assert model.set_probing.call_args_list == expected
        assert benchmark.evaluate.call_count == len(expected)

    def test_selected_layers_and_modules(self, tmp_path: Path) -> None:
        r"""Selected layers and modules, e.g. the final norm output, should be probed as given."""
        model, benchmark = _run(tmp_path, layers=[-1], modules=["rc2norm"])
        assert model.set_probing.call_args_list == [call(layer=-1, module="rc2norm")]
        assert benchmark.evaluate.call_count == 1

    def test_model_and_benchmark_configs(self, tmp_path: Path) -> None:
        r"""The model and the MTEB benchmark should be built from the evaluation config."""
        with (
            patch.object(app, "SAVING_DIR", tmp_path / "savings"),
            patch.object(app, "build_model", return_value=MagicMock(n_layers=1)) as build_model,
            patch.object(app, "build_mteb", return_value=MagicMock()) as build_mteb,
        ):
            app.eval(load_config(app.EvaluationConfig, CONFIG_DIR / "language/configs/qwen3.yaml", device="cpu"))
        assert build_model.call_args.kwargs["config"] == {"model_name": "qwen3", "model_size": "0.6B", "device": "cpu"}
        assert build_mteb.call_args.kwargs["config"]["benchmark_name"] == "Multilingual"


# -------------------------------------------------------------------
# Run directory
# -------------------------------------------------------------------
class TestLogDir:
    r"""Tests for the directory where the MTEB results are saved."""

    def test_default_log_dir(self, tmp_path: Path) -> None:
        r"""By default, results should be saved in savings/runs, as for the results of the paper."""
        assert app.EvaluationConfig().log_dir == "runs"
        _, benchmark = _run(tmp_path)
        assert benchmark.evaluate.call_args.kwargs["cache_path"] == tmp_path / "savings/runs/mteb/Qwen/Qwen3-0.6B"

    def test_custom_log_dir(self, tmp_path: Path) -> None:
        r"""A given log_dir should be used relative to savings/."""
        _, benchmark = _run(tmp_path, log_dir="ablations/final_norm")
        cache_path = benchmark.evaluate.call_args.kwargs["cache_path"]
        assert cache_path == tmp_path / "savings/ablations/final_norm/mteb/Qwen/Qwen3-0.6B"


# -------------------------------------------------------------------
# Command line interface
# -------------------------------------------------------------------
def test_main_merges_configuration_file_and_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Command line arguments should override the configuration file, which overrides the defaults."""
    config_path = CONFIG_DIR / "language/configs/qwen3.yaml"
    argv = ["eval", f"config={config_path}", "model_size=1.7B-Base", "layers=[-1]", "log_dir=ablations/qwen3_base"]
    monkeypatch.setattr("sys.argv", argv)
    with patch.object(app, "setup_logging"), patch.object(app, "eval") as run:
        app.main()
    config = run.call_args.kwargs["config"]
    assert (config.model_name, config.model_size, config.layers) == ("qwen3", "1.7B-Base", [-1])
    assert config.log_dir == "ablations/qwen3_base"
