r"""
MTEB benchmark instantiation factory.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
from typing import Any

from ...infra import build_with_type_check
from .mteb import MTEB, MTEBConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
#   Build MTEB
# ------------------------------------------------------------------------------
def build_mteb(
    config: dict[str, Any],
) -> MTEB:
    r"""
    Build MTEB benchmark.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration details.

    Returns
    -------
    benchmark: MTEB
        An instance of the specified MTEB benchmark.
    """

    config_obj = build_with_type_check(MTEBConfig, config)
    benchmark = MTEB(config_obj)
    return benchmark
