r"""
Experiment setup unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
import os

import pytest

from apps.setup import setup_logging


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
def test_setup_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""setup_logging should log at INFO level, silence noisy libraries and tokenizers parallelism warnings."""
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    monkeypatch.setattr(root, "level", root.level)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    for name in ["transformers", "datasets", "httpx", "filelock"]:
        monkeypatch.setattr(logging.getLogger(name), "level", logging.getLogger(name).level)

    setup_logging()

    assert root.level == logging.INFO
    assert all(logging.getLogger(name).level == logging.ERROR for name in ["transformers", "datasets"])
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
